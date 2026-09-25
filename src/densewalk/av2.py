"""Argoverse 2 Sensor dataset adapter.

Schemas below were read off the real downloaded feather files, not assumed:
  calibration/intrinsics.feather      -> sensor_name, fx_px, fy_px, cx_px,
                                         cy_px, k1, k2, k3, height_px, width_px
  calibration/egovehicle_SE3_sensor   -> sensor_name, qw..qz, tx_m..tz_m
  city_SE3_egovehicle.feather         -> timestamp_ns, qw..qz, tx_m..tz_m
  annotations.feather                 -> timestamp_ns, track_uuid, category,
                                         length_m, width_m, height_m, q*, t*

AV2 is already metric and gravity-aligned with Z up, so it matches the
project's canonical frame directly; no axis remap is applied. All pose
composition goes through frames.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from densewalk import frames

MASK_CATEGORIES = (
    "REGULAR_VEHICLE",
    "PEDESTRIAN",
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "BUS",
    "LARGE_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "BOX_TRUCK",
    "DOG",
    "STROLLER",
    "WHEELCHAIR",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
)

RING_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)


def _quat_trans_to_pose(qw, qx, qy, qz, tx, ty, tz) -> frames.Pose:
    """Build a Pose from AV2's (qw,qx,qy,qz) + translation convention."""
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return frames.Pose(R, np.array([tx, ty, tz], dtype=np.float64))


@dataclass(frozen=True)
class CameraModel:
    name: str
    intr: frames.Intrinsics
    ego_se3_cam: frames.Pose  # camera -> egovehicle


class Av2Sequence:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"no such AV2 sequence dir: {self.root}")
        self._intr = pd.read_feather(self.root / "calibration" / "intrinsics.feather")
        self._extr = pd.read_feather(
            self.root / "calibration" / "egovehicle_SE3_sensor.feather"
        )
        self._poses = pd.read_feather(self.root / "city_SE3_egovehicle.feather")
        self._poses = self._poses.sort_values("timestamp_ns").reset_index(drop=True)
        ann_path = self.root / "annotations.feather"
        self.annotations = pd.read_feather(ann_path) if ann_path.exists() else None

    def camera(self, name: str) -> CameraModel:
        i = self._intr[self._intr.sensor_name == name]
        if len(i) != 1:
            raise KeyError(f"camera {name!r} not in intrinsics.feather")
        i = i.iloc[0]
        e = self._extr[self._extr.sensor_name == name]
        if len(e) != 1:
            raise KeyError(f"camera {name!r} not in egovehicle_SE3_sensor.feather")
        e = e.iloc[0]
        # AV2 distortion is radial-only (k1,k2,k3); OpenCV order is
        # (k1,k2,p1,p2,k3) so the tangential terms are explicitly zero.
        dist = np.array([i.k1, i.k2, 0.0, 0.0, i.k3], dtype=np.float64)
        intr = frames.Intrinsics(
            fx=float(i.fx_px),
            fy=float(i.fy_px),
            cx=float(i.cx_px),
            cy=float(i.cy_px),
            width=int(i.width_px),
            height=int(i.height_px),
            dist=dist,
        )
        return CameraModel(
            name=name,
            intr=intr,
            ego_se3_cam=_quat_trans_to_pose(e.qw, e.qx, e.qy, e.qz, e.tx_m, e.ty_m, e.tz_m),
        )

    def image_timestamps(self, cam: str) -> list[int]:
        d = self.root / "sensors" / "cameras" / cam
        return sorted(int(p.stem) for p in d.glob("*.jpg"))

    def image_path(self, cam: str, timestamp_ns: int) -> Path:
        return self.root / "sensors" / "cameras" / cam / f"{timestamp_ns}.jpg"

    def city_se3_ego(self, timestamp_ns: int) -> frames.Pose:
        """Ego pose in city frame, nearest-neighbour in time."""
        ts = self._poses.timestamp_ns.to_numpy()
        idx = int(np.argmin(np.abs(ts - timestamp_ns)))
        r = self._poses.iloc[idx]
        return _quat_trans_to_pose(r.qw, r.qx, r.qy, r.qz, r.tx_m, r.ty_m, r.tz_m)

    def world_to_cam(self, cam: CameraModel, timestamp_ns: int) -> frames.Pose:
        """city -> camera, composed through frames.py (no inline transforms)."""
        city_se3_ego = self.city_se3_ego(timestamp_ns)
        city_se3_cam = city_se3_ego.compose(cam.ego_se3_cam)
        return city_se3_cam.inverse()

    def cuboids_at(self, timestamp_ns: int, categories=MASK_CATEGORIES) -> pd.DataFrame:
        if self.annotations is None:
            return pd.DataFrame()
        a = self.annotations
        ts = a.timestamp_ns.to_numpy()
        uniq = np.unique(ts)
        nearest = uniq[int(np.argmin(np.abs(uniq - timestamp_ns)))]
        sel = a[(a.timestamp_ns == nearest) & (a.category.isin(categories))]
        return sel

    def moving_track_ids(self, min_disp_m: float = 1.5) -> set:
        """Track ids that actually move, measured in the CITY frame.

        Cuboid centres in annotations.feather are in the egovehicle frame,
        so a parked car appears to travel the length of the drive unless
        the ego pose is composed out first. Over a 16s clip a parked car is
        static background -- masking it leaves an unsupervised hole.
        """
        if self.annotations is None:
            return set()
        per_ts = []
        for ts, g in self.annotations.groupby("timestamp_ns"):
            pose = self.city_se3_ego(int(ts))
            xyz = pose.apply(g[["tx_m", "ty_m", "tz_m"]].to_numpy())
            per_ts.append(pd.DataFrame({"tid": g.track_uuid.to_numpy(),
                                        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}))
        allpts = pd.concat(per_ts)
        moving = set()
        for tid, g in allpts.groupby("tid"):
            xyz = g[["x", "y", "z"]].to_numpy()
            if float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) > min_disp_m:
                moving.add(tid)
        return moving

    @staticmethod
    def cuboid_corners_ego(row) -> np.ndarray:
        """8 corners of an AV2 cuboid, in the egovehicle frame."""
        l, w, h = float(row.length_m), float(row.width_m), float(row.height_m)
        x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
        y = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * w / 2
        z = np.array([1, -1, 1, -1, 1, -1, 1, -1]) * h / 2
        corners_obj = np.stack([x, y, z], axis=1)
        pose = _quat_trans_to_pose(
            row.qw, row.qx, row.qy, row.qz, row.tx_m, row.ty_m, row.tz_m
        )
        return pose.apply(corners_obj)


def assert_av2_frame_sane(seq: Av2Sequence, cam_name: str = "ring_front_center") -> None:
    """Assert-on-load: AV2 must be metric, Z-up, with a plausible camera
    mount height above the road."""
    cam = seq.camera(cam_name)
    mount_z = float(cam.ego_se3_cam.t[2])
    if not (0.5 <= mount_z <= 3.0):
        raise AssertionError(
            f"{cam_name} mount height {mount_z:.3f}m off the egovehicle origin "
            "is implausible -- units or axis convention wrong"
        )
    ts = seq.image_timestamps(cam_name)
    if len(ts) < 2:
        raise AssertionError(f"{cam_name} has {len(ts)} frames, need >= 2")
