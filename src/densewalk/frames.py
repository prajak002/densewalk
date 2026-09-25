"""Canonical coordinate frame for the project.

World convention (non-negotiable, see project invariants):
    - metres
    - Z-up
    - floor at z=0
    - right-handed

ALL coordinate conversions (camera extrinsics, dataset-specific ground
planes, axis remaps, unit conversions) go through this module. No other
module should write inline rotation matrices, axis swaps, or unit scale
factors -- import the primitives here instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

FLOOR_Z = 0.0


@dataclass(frozen=True)
class Pose:
    """A rigid transform. Applying it to a point in frame A yields the
    point's coordinates in frame B, i.e. p_B = R @ p_A + t."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"Pose.R must be (3,3), got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            raise ValueError("Pose.R is not orthonormal (not a valid rotation)")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
            raise ValueError("Pose.R has determinant != 1 (improper rotation)")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(T: np.ndarray) -> "Pose":
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"expected 4x4 homogeneous matrix, got {T.shape}")
        return Pose(T[:3, :3], T[:3, 3])

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points from frame A into frame B."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return points @ self.R.T + self.t

    def inverse(self) -> "Pose":
        R_inv = self.R.T
        return Pose(R_inv, -R_inv @ self.t)

    def compose(self, other: "Pose") -> "Pose":
        """self ∘ other: apply `other` first, then `self`."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


@dataclass(frozen=True)
class AxisRemap:
    """Declarative axis permutation + sign flip, e.g. to turn a Y-up,
    dataset-native frame into the project's canonical Z-up frame.

    `source_axis_for` maps each *output* axis to the source axis and sign
    that fills it. E.g. converting Y-up (X right, Y up, Z backward-ish) to
    Z-up (X right, Y forward, Z up) is:
        out_x = src_x  -> (Axis.X, +1)
        out_y = -src_z -> (Axis.Z, -1)
        out_z = src_y  -> (Axis.Y, +1)
    """

    out_x: tuple[Axis, float]
    out_y: tuple[Axis, float]
    out_z: tuple[Axis, float]

    def matrix(self) -> np.ndarray:
        M = np.zeros((3, 3))
        for row, (src_axis, sign) in enumerate((self.out_x, self.out_y, self.out_z)):
            M[row, src_axis.value] = sign
        if not np.allclose(M.T @ M, np.eye(3), atol=1e-9):
            raise ValueError("AxisRemap does not define an orthonormal matrix")
        return M

    def as_pose(self) -> Pose:
        return Pose(self.matrix(), np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.as_pose().apply(points)


IDENTITY_REMAP = AxisRemap((Axis.X, 1.0), (Axis.Y, 1.0), (Axis.Z, 1.0))
# Y-up, right-handed (X right, Y up, Z toward viewer) -> project canonical
# Z-up, right-handed (X right, Y forward, Z up).
YUP_TO_ZUP = AxisRemap((Axis.X, 1.0), (Axis.Z, -1.0), (Axis.Y, 1.0))


def cm_to_m(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x, dtype=np.float64) / 100.0 if isinstance(x, np.ndarray) else x / 100.0


def mm_to_m(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x, dtype=np.float64) / 1000.0 if isinstance(x, np.ndarray) else x / 1000.0


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics, OpenCV convention (X right, Y down,
    Z forward out of the camera)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray  # OpenCV distortion coeffs (k1,k2,p1,p2[,k3...]), may be all-zero

    def __post_init__(self) -> None:
        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "dist", dist)

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


def opencv_rt_to_world_to_cam_pose(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    """OpenCV calibration (e.g. solvePnP, checkerboard extrinsics) gives
    rvec/tvec such that p_cam = R @ p_world + t, in the *dataset's native*
    world frame -- not necessarily our canonical Z-up/floor-zero frame.
    This returns that raw world->camera Pose; canonicalize the dataset's
    world frame separately (see AxisRemap) before composing.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return Pose(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def colmap_qvec_to_world_to_cam_pose(qvec: np.ndarray, tvec: np.ndarray) -> Pose:
    """COLMAP's images.txt stores each image as QW QX QY QZ TX TY TZ, which
    is the world->camera rotation as a Hamilton quaternion plus translation,
    i.e. p_cam = R(q) @ p_world + t.

    COLMAP's world frame is arbitrary: Y points roughly down (it inherits the
    OpenCV camera convention) and the scale is unitless, so the result is NOT
    in our canonical metric Z-up frame. Canonicalize separately -- see
    YUP_TO_ZUP for the axis remap and assert_metric_scale for the scale.
    """
    w, x, y, z = (float(v) for v in np.asarray(qvec, dtype=np.float64).reshape(4))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("COLMAP qvec has zero norm")
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return Pose(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def world_to_pixel(points_world: np.ndarray, world_to_cam: Pose, intr: Intrinsics) -> np.ndarray:
    """Project canonical-world (N,3) points to (N,2) pixel coordinates."""
    points_cam = world_to_cam.apply(points_world)
    rvec, _ = cv2.Rodrigues(np.eye(3))
    pix, _ = cv2.projectPoints(
        points_cam.reshape(-1, 1, 3),
        rvec,
        np.zeros(3),
        intr.K(),
        intr.dist,
    )
    return pix.reshape(-1, 2)


def pixel_to_ray_cam(pixels: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Undistort + unproject (N,2) pixels to (N,3) unit rays in the camera
    frame (OpenCV convention, +Z forward)."""
    pixels = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    undist = cv2.undistortPoints(pixels.reshape(-1, 1, 2), intr.K(), intr.dist)
    undist = undist.reshape(-1, 2)
    rays = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def is_in_front_of_camera(points_world: np.ndarray, world_to_cam: Pose) -> np.ndarray:
    points_cam = world_to_cam.apply(points_world)
    return points_cam[:, 2] > 0


def assert_floor_at_zero(
    floor_points: np.ndarray, tol_m: float = 0.05, min_points: int = 3
) -> None:
    """Sanity assertion: a set of points known to lie on the walking
    surface must sit within `tol_m` of z=0 in canonical world coordinates.
    """
    pts = np.atleast_2d(np.asarray(floor_points, dtype=np.float64))
    if pts.shape[0] < min_points:
        raise ValueError(f"need >= {min_points} floor points to assert, got {pts.shape[0]}")
    z = pts[:, 2]
    median_z = float(np.median(z))
    if abs(median_z - FLOOR_Z) > tol_m:
        raise AssertionError(
            f"floor points have median z={median_z:.4f}m, expected ~{FLOOR_Z}m (tol={tol_m}m)"
        )


def assert_metric_scale(
    points_world: np.ndarray, expected_extent_m: tuple[float, float], axis: int = 2
) -> None:
    """Sanity assertion against unit-conversion bugs (e.g. forgetting a
    cm->m or mm->m conversion): the scene's extent along `axis` must fall
    within a plausible metric range, e.g. a person's height in [0.3, 2.5]m
    or a room extent in [1, 100]m."""
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    extent = float(pts[:, axis].max() - pts[:, axis].min())
    lo, hi = expected_extent_m
    if not (lo <= extent <= hi):
        raise AssertionError(
            f"extent along axis {axis} is {extent:.4f}m, expected within [{lo}, {hi}]m "
            "-- check for a missing unit conversion"
        )


# ---------------------------------------------------------------------------
# Planar (SE(2)) ego motion, used by the Δt horizon dataset.
#
# Ego frame at time t: x forward, y LEFT, z up (the canonical Z-up right-handed
# frame, viewed from above). A planar pose is (x, y, yaw) of the ego in a
# clip-local world frame whose origin is the clip's first labelled frame.
#
# DenseWalk labels store `direction_deg` = motion heading relative to camera
# forward with NEGATIVE = LEFT (verified in the DCA notebook: stepping_left is
# 100% negative dir_deg). In our y-left frame that is angle = -direction_deg.
# ---------------------------------------------------------------------------

DENSEWALK_DIR_SIGN = -1.0  # dir_deg (negative=left) -> CCW angle in x-fwd/y-left frame


def label_dir_to_angle(direction_deg: np.ndarray | float, sign: float = DENSEWALK_DIR_SIGN):
    """DenseWalk direction_deg -> motion angle (rad), CCW-positive in the ego frame."""
    return np.radians(np.asarray(direction_deg, dtype=np.float64)) * sign


def integrate_planar_odometry(
    t: np.ndarray, v: np.ndarray, motion_angle: np.ndarray, yaw_rate: np.ndarray
) -> np.ndarray:
    """Dead-reckon (N,3) planar poses [x, y, yaw] from per-sample speed (m/s),
    motion angle relative to body forward (rad, CCW+) and yaw rate (rad/s,
    CCW+). Zero-order hold: sample k's values apply over [t_k, t_{k+1})."""
    t = np.asarray(t, dtype=np.float64)
    n = t.shape[0]
    P = np.zeros((n, 3))
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        if dt <= 0:
            raise ValueError(f"timestamps not strictly increasing at index {k}: {t[k]} -> {t[k+1]}")
        h = P[k, 2] + motion_angle[k]
        P[k + 1, 0] = P[k, 0] + v[k] * dt * math.cos(h)
        P[k + 1, 1] = P[k, 1] + v[k] * dt * math.sin(h)
        P[k + 1, 2] = P[k, 2] + yaw_rate[k] * dt
    return P


def se2_interp(P0: np.ndarray, P1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate planar poses; yaw along the shortest arc."""
    dyaw = math.atan2(math.sin(P1[2] - P0[2]), math.cos(P1[2] - P0[2]))
    return np.array([P0[0] + alpha * (P1[0] - P0[0]),
                     P0[1] + alpha * (P1[1] - P0[1]),
                     P0[2] + alpha * dyaw])


def se2_relative_xy(P_ref: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Position of pose P expressed in the ego frame of P_ref (x fwd, y left), metres."""
    c, s = math.cos(P_ref[2]), math.sin(P_ref[2])
    dx, dy = P[0] - P_ref[0], P[1] - P_ref[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy])
