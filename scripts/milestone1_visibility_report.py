"""Milestone 1 artifact: static visibility fraction per background region.

For one reconstruction sequence (WildTrack), computes, for a sampled grid
of ground-plane background points, the fraction of (camera, frame)
observation pairs in which each point is visible (unoccluded by any
person) -- the "view-level" metric, which gives continuous resolution
over camera_count * frame_count observations per point, unlike the
coarser "visible from >=1 camera per frame" (frame-level) metric, which
is also reported for context but is NOT the headline number.

Person body radius is a free parameter this dataset doesn't give us
(WildTrack has ground-plane positions, not body extent) -- so the
result is swept over a plausible radius range and both the mean curve
and its stability are reported, instead of picking one radius silently.

Gate: refuses to run unless scripts/verify_calibration.py has already
produced its overlay PNGs in --calib-check-dir (or --force is passed).
Unit tests cannot catch a wrong coordinate convention or dataset schema
assumption; the calibration overlay can.

Usage:
    uv run python scripts/verify_calibration.py --data-dir data/wildtrack/extracted
    # inspect artifacts/calibration_check/*.png, THEN:
    uv run python scripts/milestone1_visibility_report.py \
        --data-dir data/wildtrack/extracted \
        --out artifacts/visibility_fraction_fig.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from densewalk import frames, visibility, wildtrack

NUM_CAMERAS = 7
RADIUS_SWEEP_M = (0.20, 0.25, 0.30, 0.35)


def find_calibration_files(calib_dir: Path) -> list[tuple[Path, Path]]:
    intrinsic_dir = calib_dir / "intrinsic_zero"
    extrinsic_dir = calib_dir / "extrinsic"
    if not intrinsic_dir.is_dir() or not extrinsic_dir.is_dir():
        raise FileNotFoundError(
            f"expected {intrinsic_dir} and {extrinsic_dir} to exist -- "
            "check the extracted WildTrack directory layout, it may not "
            "match what this script assumes"
        )
    intrinsics = sorted(intrinsic_dir.glob("*.xml"))
    extrinsics = sorted(extrinsic_dir.glob("*.xml"))
    if len(intrinsics) != NUM_CAMERAS or len(extrinsics) != NUM_CAMERAS:
        raise FileNotFoundError(
            f"expected {NUM_CAMERAS} intrinsic and {NUM_CAMERAS} extrinsic "
            f"files, found {len(intrinsics)} and {len(extrinsics)} in {calib_dir}. "
            "Not guessing camera correspondence -- inspect the directory."
        )
    return list(zip(intrinsics, extrinsics))


def require_calibration_verified(calib_check_dir: Path, force: bool) -> None:
    overlays = sorted(calib_check_dir.glob("*overlay.png")) if calib_check_dir.is_dir() else []
    if len(overlays) >= NUM_CAMERAS or force:
        if force and len(overlays) < NUM_CAMERAS:
            print(f"WARNING: --force set, proceeding without {NUM_CAMERAS} calibration "
                  f"overlays (found {len(overlays)}). Numbers below are UNVERIFIED.")
        return
    raise RuntimeError(
        f"calibration not verified: expected >= {NUM_CAMERAS} overlay PNGs in "
        f"{calib_check_dir}, found {len(overlays)}. Run scripts/verify_calibration.py "
        "first and visually inspect the overlays, or pass --force to override."
    )


def load_cameras(calib_dir: Path, img_width: int, img_height: int) -> list[np.ndarray]:
    centers = []
    for intr_path, extr_path in find_calibration_files(calib_dir):
        intr, world_to_cam = wildtrack.load_camera(
            str(intr_path), str(extr_path), img_width, img_height
        )
        wildtrack.assert_camera_pose_plausible(world_to_cam)
        center = world_to_cam.inverse().apply(np.zeros((1, 3)))[0]
        centers.append(center)
        print(f"  camera {intr_path.stem}: center={center.round(2).tolist()}")
    return centers


def load_frame_positions(annotations_dir: Path) -> list[np.ndarray]:
    """Parse annotation JSON once into per-frame (M,2) ground positions
    (x,y). Kept separate from FrameOccupants construction so the radius
    sweep doesn't re-parse JSON for every radius value."""
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no annotation JSON files found in {annotations_dir}")

    per_frame = []
    for f in files:
        entries = json.loads(f.read_text())
        position_ids = []
        for entry in entries:
            if "positionID" not in entry:
                raise KeyError(
                    f"annotation entry in {f} has no 'positionID' field -- "
                    f"got keys {list(entry.keys())}, schema assumption was wrong"
                )
            position_ids.append(entry["positionID"])
        if position_ids:
            world_pts = wildtrack.position_id_to_world(np.array(position_ids))
            per_frame.append(world_pts[:, :2])
        else:
            per_frame.append(np.zeros((0, 2)))
    return per_frame


def build_frame_occupants(
    per_frame_positions: list[np.ndarray], radius: float, height: float
) -> list[visibility.FrameOccupants]:
    return [
        visibility.FrameOccupants(
            people=[
                visibility.Person(id=i, x=float(p[0]), y=float(p[1]), radius=radius, height=height)
                for i, p in enumerate(positions)
            ]
        )
        for positions in per_frame_positions
    ]


def sample_background_grid(stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = np.arange(0, wildtrack.GRID_WIDTH, stride)
    y_idx = np.arange(0, wildtrack.GRID_HEIGHT, stride)
    xx, yy = np.meshgrid(x_idx, y_idx, indexing="xy")
    ids = (yy * wildtrack.GRID_WIDTH + xx).ravel()
    points = wildtrack.position_id_to_world(ids)
    return points, xx, yy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/wildtrack/extracted"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/visibility_fraction_fig.png"))
    ap.add_argument("--calib-check-dir", type=Path, default=Path("artifacts/calibration_check"))
    ap.add_argument("--force", action="store_true", help="skip the calibration-verified gate")
    ap.add_argument("--grid-stride", type=int, default=15, help="ground-grid subsample stride")
    ap.add_argument("--img-width", type=int, default=1920)
    ap.add_argument("--img-height", type=int, default=1080)
    ap.add_argument("--person-height", type=float, default=visibility.DEFAULT_PERSON_HEIGHT_M)
    args = ap.parse_args()

    require_calibration_verified(args.calib_check_dir, args.force)

    calib_dir = args.data_dir / "calibrations"
    annotations_dir = args.data_dir / "annotations_positions"

    print(f"Loading cameras from {calib_dir} ...")
    cams = load_cameras(calib_dir, args.img_width, args.img_height)

    print(f"Loading annotated frame positions from {annotations_dir} ...")
    per_frame_positions = load_frame_positions(annotations_dir)
    n_frames = len(per_frame_positions)
    avg_people = np.mean([len(p) for p in per_frame_positions])
    total_pairs = len(cams) * n_frames
    print(f"  {n_frames} frames x {len(cams)} cameras = {total_pairs} (camera,frame) observations, "
          f"{avg_people:.1f} people/frame avg")

    print(f"Sampling background grid at stride={args.grid_stride} ...")
    points, xx, yy = sample_background_grid(args.grid_stride)
    print(f"  {points.shape[0]} background points")
    frames.assert_metric_scale(points, expected_extent_m=(5.0, 40.0), axis=1)
    frames.assert_floor_at_zero(points)

    print(f"Sweeping person radius over {RADIUS_SWEEP_M} m to check result stability ...")
    sweep_frac = {}
    for radius in RADIUS_SWEEP_M:
        frame_list = build_frame_occupants(per_frame_positions, radius, args.person_height)
        view_level = visibility.static_visibility_fraction_view_level_batch(points, cams, frame_list)
        sweep_frac[radius] = view_level
        print(f"  radius={radius:.2f}m: mean={view_level.mean():.3f}  "
              f"frac(<0.05)={np.mean(view_level < 0.05):.3f}")

    default_radius = 0.25
    view_level = sweep_frac[default_radius]
    frame_list_default = build_frame_occupants(per_frame_positions, default_radius, args.person_height)
    frame_level = visibility.static_visibility_fraction_batch(points, cams, frame_list_default)

    grid_view = view_level.reshape(xx.shape)
    never_unoccluded = float(np.mean(view_level < 0.05))
    print(f"\n[radius={default_radius}m] view-level mean: {view_level.mean():.3f}, "
          f"frame-level mean: {frame_level.mean():.3f} (these differ on purpose, see docstring)")
    print(f"  fraction of area with view-level visibility < 0.05: {never_unoccluded:.3f}")

    mean_by_radius = [sweep_frac[r].mean() for r in RADIUS_SWEEP_M]
    radius_range = max(mean_by_radius) - min(mean_by_radius)
    print(f"  mean visibility range across radius sweep: {radius_range:.3f} "
          f"({'STABLE' if radius_range < 0.05 else 'RADIUS-SENSITIVE -- report this'})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax0, ax1, ax2, ax3 = axes.ravel()

    x_m = points[:, 0].reshape(xx.shape)
    y_m = points[:, 1].reshape(xx.shape)
    im = ax0.pcolormesh(x_m, y_m, grid_view, shading="auto", cmap="viridis", vmin=0, vmax=1)
    cam_arr = np.array(cams)
    ax0.scatter(cam_arr[:, 0], cam_arr[:, 1], c="red", marker="^", s=80, label="cameras")
    ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title(f"View-level visibility fraction per region\n(radius={default_radius}m, "
                  f"{total_pairs} camera-frame observations)")
    ax0.legend(loc="upper right"); ax0.set_aspect("equal")
    fig.colorbar(im, ax=ax0, label="fraction of (camera,frame) pairs visible")

    ax1.hist(view_level, bins=40, color="steelblue", alpha=0.7, label="view-level")
    ax1.hist(frame_level, bins=40, color="darkorange", alpha=0.5, label="frame-level (any camera)")
    ax1.set_xlabel("visibility fraction"); ax1.set_ylabel("# background regions")
    ax1.set_title(f"Distribution: view-level vs frame-level\n({never_unoccluded * 100:.1f}% of area "
                  "< 0.05 view-level visible")
    ax1.legend()

    for radius in RADIUS_SWEEP_M:
        ax2.hist(sweep_frac[radius], bins=40, alpha=0.5, label=f"r={radius:.2f}m")
    ax2.set_xlabel("view-level visibility fraction"); ax2.set_ylabel("# background regions")
    ax2.set_title("Radius sweep: full distribution")
    ax2.legend()

    ax3.plot(RADIUS_SWEEP_M, mean_by_radius, "o-")
    ax3.set_xlabel("assumed person radius (m)"); ax3.set_ylabel("mean view-level visibility fraction")
    ax3.set_title(f"Headline number vs. radius assumption\nrange={radius_range:.3f} "
                  f"({'stable' if radius_range < 0.05 else 'SENSITIVE'})")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    sys.exit(main())
