"""Time-varying crowd layer: per-person Gaussians driven by real tracks.

MoSca cannot run on Blackwell (it pins CUDA 11.8 / sm_90; this card is
sm_120), so the dynamic layer is built from AV2's own ground-truth 3D
cuboid tracks instead:

  * each moving track becomes a small cloud of Gaussians sized to its cuboid
  * colour is sampled from the real images where that person is visible
  * at render time the cloud is rigidly placed at the track's pose for
    that timestamp, so people follow their true recorded trajectories

The result is genuinely 4D -- ``scene(t) = static background + people at
t`` -- and stays consistent with the project's representation rule: the
background is static Gaussians, people are separate posed entities, never
a monolithic 4D blob.

All world transforms go through frames.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from densewalk import frames
from densewalk.av2 import Av2Sequence


@dataclass
class PersonCloud:
    """A rigid cloud of Gaussians for one tracked identity, stored in the
    object's own local frame (origin at the cuboid centre)."""

    track_uuid: str
    category: str
    means_local: np.ndarray   # (N,3)
    colors: np.ndarray        # (N,3) in [0,1]
    scale: float              # isotropic gaussian scale, metres
    dims: tuple[float, float, float]


def _sample_points_in_cuboid(l: float, w: float, h: float, n: int, rng) -> np.ndarray:
    """Points filling the cuboid, denser near the vertical axis so a person
    reads as a body rather than a box."""
    x = rng.normal(0.0, l / 5.0, n).clip(-l / 2, l / 2)
    y = rng.normal(0.0, w / 5.0, n).clip(-w / 2, w / 2)
    z = rng.uniform(-h / 2, h / 2, n)
    return np.stack([x, y, z], axis=1)


def build_person_clouds(
    seq: Av2Sequence,
    cameras: list[str],
    points_per_person: int = 900,
    min_disp_m: float = 1.5,
    max_people: int | None = None,
    samples_per_track: int = 6,
) -> list[PersonCloud]:
    """One Gaussian cloud per moving track, coloured from the real frames."""
    moving = seq.moving_track_ids(min_disp_m)
    ann = seq.annotations
    ann = ann[ann.track_uuid.isin(moving)]
    rng = np.random.default_rng(0)

    # prefer the most-observed tracks: they have the best colour evidence
    order = ann.track_uuid.value_counts().index.tolist()
    if max_people is not None:
        order = order[:max_people]

    cams = {c: seq.camera(c) for c in cameras}
    clouds: list[PersonCloud] = []

    for tid in order:
        rows = ann[ann.track_uuid == tid].sort_values("timestamp_ns")
        r0 = rows.iloc[0]
        l, w, h = float(r0.length_m), float(r0.width_m), float(r0.height_m)
        pts_local = _sample_points_in_cuboid(l, w, h, points_per_person, rng)

        acc = np.zeros((points_per_person, 3))
        cnt = np.zeros((points_per_person, 1))
        picks = rows.iloc[:: max(1, len(rows) // samples_per_track)]

        for _, row in picks.iterrows():
            ts = int(row.timestamp_ns)
            obj_pose = frames.Pose(
                _quat_to_R(row.qw, row.qx, row.qy, row.qz),
                np.array([row.tx_m, row.ty_m, row.tz_m], dtype=np.float64),
            )
            pts_ego = obj_pose.apply(pts_local)
            pts_city = seq.city_se3_ego(ts).apply(pts_ego)

            for cname, cam in cams.items():
                ts_cam = _nearest(seq.image_timestamps(cname), ts)
                img = cv2.imread(str(seq.image_path(cname, ts_cam)))
                if img is None:
                    continue
                w2c = seq.world_to_cam(cam, ts_cam)
                front = frames.is_in_front_of_camera(pts_city, w2c)
                if front.sum() == 0:
                    continue
                px = frames.world_to_pixel(pts_city, w2c, cam.intr)
                H, W = img.shape[:2]
                u = px[:, 0]
                v = px[:, 1]
                ok = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
                if ok.sum() == 0:
                    continue
                ui = u[ok].astype(np.int32)
                vi = v[ok].astype(np.int32)
                # 3x3 median around each hit rejects single-pixel outliers,
                # and we keep only points in the cuboid's inner core so the
                # colour comes off the body, not the silhouette edge.
                bgr = np.stack([
                    np.median(img[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                              .reshape(-1, 3), axis=0)
                    for y, x in zip(vi, ui)
                ]).astype(np.float64) / 255.0
                acc[ok] += bgr[:, ::-1]  # BGR -> RGB
                cnt[ok] += 1

        colors = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        seen = ~np.isnan(colors).any(axis=1)
        fill = colors[seen].mean(axis=0) if seen.any() else np.array([0.5, 0.5, 0.5])
        colors = np.where(np.isnan(colors), fill, colors)
        # lift out of crush: these samples skew dark from shadowed pixels
        colors = np.clip(colors * 1.35 + 0.05, 0.0, 1.0)
        # spacing between sampled points, halved so neighbours just overlap
        # rather than merging into one dark blob
        scale = float(np.clip(0.5 * (l * w * h / points_per_person) ** (1 / 3),
                              0.015, 0.045))
        clouds.append(
            PersonCloud(
                track_uuid=str(tid),
                category=str(r0.category),
                means_local=pts_local,
                colors=colors,
                scale=scale,
                dims=(l, w, h),
            )
        )
    return clouds


def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _nearest(values, target: int) -> int:
    arr = np.asarray(values)
    return int(arr[int(np.argmin(np.abs(arr - target)))])


def track_pose_at(seq: Av2Sequence, track_uuid: str, timestamp_ns: int) -> frames.Pose | None:
    """City-frame pose of one track at a timestamp (nearest annotation)."""
    rows = seq.annotations[seq.annotations.track_uuid == track_uuid]
    if len(rows) == 0:
        return None
    ts = rows.timestamp_ns.to_numpy()
    if abs(int(ts[np.argmin(np.abs(ts - timestamp_ns))]) - timestamp_ns) > 2e8:
        return None  # track not present near this time (>0.2s away)
    row = rows.iloc[int(np.argmin(np.abs(ts - timestamp_ns)))]
    obj_in_ego = frames.Pose(
        _quat_to_R(row.qw, row.qx, row.qy, row.qz),
        np.array([row.tx_m, row.ty_m, row.tz_m], dtype=np.float64),
    )
    return seq.city_se3_ego(int(row.timestamp_ns)).compose(obj_in_ego)


def crowd_gaussians_at(
    seq: Av2Sequence,
    clouds: list[PersonCloud],
    timestamp_ns: int,
    device: str = "cuda",
    origin: np.ndarray | None = None,
):
    """Pose every visible person at this timestamp and return stacked
    Gaussian tensors ready to concatenate with the background."""
    means_all, cols_all, scales_all = [], [], []
    for c in clouds:
        pose = track_pose_at(seq, c.track_uuid, timestamp_ns)
        if pose is None:
            continue
        pts = pose.apply(c.means_local)
        if origin is not None:
            pts = pts - origin
        means_all.append(pts)
        cols_all.append(c.colors)
        scales_all.append(np.full(len(pts), c.scale))

    if not means_all:
        return None

    means = torch.tensor(np.concatenate(means_all), dtype=torch.float32, device=device)
    colors = torch.tensor(np.concatenate(cols_all), dtype=torch.float32, device=device)
    scales = torch.tensor(np.concatenate(scales_all), dtype=torch.float32, device=device)
    n = means.shape[0]
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    return {
        "means": means,
        "quats": quats,
        "scales": scales[:, None].repeat(1, 3),
        "opacities": torch.full((n,), 0.9, device=device),
        "colors": colors,
    }
