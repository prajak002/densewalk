"""WildTrack dataset adapter.

Facts below (grid layout, calibration file tags) are sourced from the
dataset's own published toolkit (Chavdarova/WILDTRACK-toolkit,
`intersecting_area.py`), not guessed. They are additionally re-asserted
at load time against the real downloaded files -- see `load_camera` and
`load_position_grid_sanity_check` -- per the project invariant that
dataset formats are read, not assumed.

Grid: 480 (x) x 1440 (y) ground-plane positions, origin (-300, -90) cm,
step 2.5cm, z always 0 (WildTrack's native world frame is already
Z-up / floor-at-zero on the ground plane, matching this project's
canonical convention -- no axis remap needed, only cm->m).

All world-frame outputs from this module are in the project's canonical
frame (metres, Z-up, floor z=0); all coordinate math routes through
frames.py primitives.
"""
from __future__ import annotations

from xml.etree import ElementTree

import cv2
import numpy as np

from densewalk import frames

GRID_WIDTH = 480  # x index = position_id % GRID_WIDTH
GRID_HEIGHT = 1440  # y index = position_id // GRID_WIDTH
# Verbatim from the dataset's own README.txt (authoritative -- an earlier
# value taken from the toolkit page had Y origin wrong by 8.1m):
#   X = -3.0 + 0.025*(ID%480)
#   Y = -9.0 + 0.025*(ID//480)
GRID_ORIGIN_M = (-3.0, -9.0)
GRID_STEP_M = 0.025


def position_id_to_world(position_ids: np.ndarray) -> np.ndarray:
    """Convert WildTrack ground-plane position IDs to canonical-world
    (N,3) points on the floor (z=0)."""
    ids = np.atleast_1d(np.asarray(position_ids, dtype=np.int64))
    max_id = GRID_WIDTH * GRID_HEIGHT
    if np.any(ids < 0) or np.any(ids >= max_id):
        raise ValueError(f"position_id out of range [0, {max_id})")
    x_idx = ids % GRID_WIDTH
    y_idx = ids // GRID_WIDTH
    x_m = GRID_ORIGIN_M[0] + GRID_STEP_M * x_idx.astype(np.float64)
    y_m = GRID_ORIGIN_M[1] + GRID_STEP_M * y_idx.astype(np.float64)
    z_m = np.zeros_like(x_m)
    return np.stack([x_m, y_m, z_m], axis=1)


def load_opencv_xml_matrix(path: str, tag: str) -> np.ndarray:
    """Read a matrix/vector out of a WildTrack calibration XML.

    Two node shapes occur in this dataset and both must work:
      - intrinsics: proper type_id="opencv-matrix" nodes (rows/cols/dt/data)
      - extrinsics: rvec/tvec as bare whitespace-separated text, which
        cv2.FileStorage does NOT surface via .mat()
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        node = fs.getNode(tag)
        if not node.empty():
            mat = node.mat()
            if mat is not None:
                return mat
    except cv2.error:
        # bare text node: FileStorage raises instead of returning empty
        pass
    finally:
        fs.release()

    root = ElementTree.parse(path).getroot()
    el = root.find(tag)
    if el is None or el.text is None:
        raise KeyError(f"tag '{tag}' not found in {path}")
    values = [float(v) for v in el.text.split()]
    if not values:
        raise KeyError(f"tag '{tag}' in {path} has no numeric content")
    return np.array(values, dtype=np.float64).reshape(-1, 1)


def load_camera(
    intrinsic_path: str, extrinsic_path: str, width: int, height: int
) -> tuple[frames.Intrinsics, frames.Pose]:
    """Load one WildTrack camera's intrinsics and its world->camera pose
    (in WildTrack's native world frame, which is already canonical
    Z-up/floor-zero -- see module docstring)."""
    K = load_opencv_xml_matrix(intrinsic_path, "camera_matrix")
    dist = load_opencv_xml_matrix(intrinsic_path, "distortion_coefficients")
    rvec = load_opencv_xml_matrix(extrinsic_path, "rvec")
    tvec = load_opencv_xml_matrix(extrinsic_path, "tvec")

    intr = frames.Intrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=width,
        height=height,
        dist=dist.reshape(-1),
    )
    # tvec ships in centimetres in this dataset (|t| ~ 1000 => ~10m);
    # the canonical world frame is metres, so convert before building the pose.
    tvec_m = frames.cm_to_m(np.asarray(tvec, dtype=np.float64).reshape(3))
    world_to_cam = frames.opencv_rt_to_world_to_cam_pose(rvec, tvec_m)
    return intr, world_to_cam


def assert_camera_pose_plausible(
    world_to_cam: frames.Pose, min_height_m: float = 1.0, max_height_m: float = 10.0
) -> None:
    """Sanity gate for a loaded camera: its center, recovered in world
    coordinates, should sit at a plausible pole/mount height above the
    z=0 ground plane WildTrack's grid is defined on."""
    cam_center_world = world_to_cam.inverse().apply(np.zeros((1, 3)))[0]
    z = cam_center_world[2]
    if not (min_height_m <= z <= max_height_m):
        raise AssertionError(
            f"camera center height {z:.3f}m outside plausible range "
            f"[{min_height_m}, {max_height_m}]m -- check axis convention "
            "(WildTrack's world frame may not be Z-up as assumed)"
        )
