"""Calibration/schema verification gate -- run this BEFORE
milestone1_visibility_report.py, not after.

Unit tests on wildtrack.py validate that the loader is internally
consistent with its own assumed schema; they cannot catch a wrong
schema or a wrong intrinsic/extrinsic <-> camera correspondence, because
the test fixtures were written against the same assumption being tested.

This script is the actual check: project each person's annotated
ground-plane position (positionID -> world, via wildtrack.py) through
the loaded calibration for each camera, and draw the reprojected point
next to that camera's own annotated bounding box on the real frame
image. If the schema, directory layout, or camera<->view correspondence
is wrong, the marker will visibly NOT land inside/near the box. If it's
right, they align. This is a visual gate -- inspect the output PNGs
before trusting milestone1_visibility_report.py's numbers.

Usage:
    uv run python scripts/verify_calibration.py \
        --data-dir data/wildtrack/extracted \
        --out-dir artifacts/calibration_check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from densewalk import wildtrack

NUM_CAMERAS = 7


def find_calibration_files(calib_dir: Path) -> list[tuple[Path, Path]]:
    intrinsic_dir = calib_dir / "intrinsic_zero"
    extrinsic_dir = calib_dir / "extrinsic"
    if not intrinsic_dir.is_dir() or not extrinsic_dir.is_dir():
        raise FileNotFoundError(f"expected {intrinsic_dir} and {extrinsic_dir} to exist")
    intrinsics = sorted(intrinsic_dir.glob("*.xml"))
    extrinsics = sorted(extrinsic_dir.glob("*.xml"))
    if len(intrinsics) != NUM_CAMERAS or len(extrinsics) != NUM_CAMERAS:
        raise FileNotFoundError(
            f"expected {NUM_CAMERAS} intrinsic/extrinsic files, "
            f"found {len(intrinsics)}/{len(extrinsics)} in {calib_dir}"
        )
    return list(zip(intrinsics, extrinsics))


def find_frame_images(data_dir: Path, frame_stem: str) -> list[Path]:
    """Search the whole extracted tree for image files matching one
    frame's stem (e.g. '00000000'), one per camera directory. We do NOT
    assume a specific images directory name -- WildTrack's own layout
    for this isn't confirmed yet, so we search and report what we find
    rather than guessing and silently loading the wrong file."""
    candidates = sorted(data_dir.rglob(f"{frame_stem}.png")) + sorted(
        data_dir.rglob(f"{frame_stem}.jpg")
    )
    if not candidates:
        raise FileNotFoundError(
            f"no image files named {frame_stem}.png/.jpg found anywhere under {data_dir}. "
            "Inspect the extracted archive and adjust this function -- do not guess the path."
        )
    by_dir = sorted({p.parent for p in candidates})
    images = []
    for d in by_dir:
        matches = sorted(d.glob(f"{frame_stem}.*"))
        images.append(matches[0])
    if len(images) != NUM_CAMERAS:
        raise FileNotFoundError(
            f"expected images for {NUM_CAMERAS} cameras, found {len(images)} directories "
            f"containing {frame_stem}.*: {[str(d) for d in by_dir]}. "
            "Camera<->directory correspondence is unconfirmed -- inspect before proceeding."
        )
    return images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/wildtrack/extracted"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/calibration_check"))
    ap.add_argument("--frame-index", type=int, default=0, help="which annotated frame to check")
    ap.add_argument("--img-width", type=int, default=1920)
    ap.add_argument("--img-height", type=int, default=1080)
    args = ap.parse_args()

    calib_dir = args.data_dir / "calibrations"
    annotations_dir = args.data_dir / "annotations_positions"

    ann_files = sorted(annotations_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"no annotation files in {annotations_dir}")
    ann_file = ann_files[args.frame_index]
    frame_stem = ann_file.stem
    print(f"Checking frame {frame_stem} ({ann_file}) ...")

    entries = json.loads(ann_file.read_text())
    print(f"  {len(entries)} annotated people in this frame")
    for e in entries[:1]:
        print(f"  sample annotation keys: {sorted(e.keys())}")
        if "views" in e:
            print(f"  sample views[0] keys: {sorted(e['views'][0].keys())}")

    calib_pairs = find_calibration_files(calib_dir)
    image_paths = find_frame_images(args.data_dir, frame_stem)
    print("Camera <-> image directory correspondence used (index order, VERIFY visually):")
    for i, ((intr_p, extr_p), img_p) in enumerate(zip(calib_pairs, image_paths)):
        print(f"  view {i}: calib={intr_p.stem}/{extr_p.stem}  image={img_p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, ((intr_path, extr_path), img_path) in enumerate(zip(calib_pairs, image_paths)):
        intr, world_to_cam = wildtrack.load_camera(
            str(intr_path), str(extr_path), args.img_width, args.img_height
        )
        wildtrack.assert_camera_pose_plausible(world_to_cam)

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"cv2 failed to read {img_path}")

        for entry in entries:
            if "positionID" not in entry:
                raise KeyError(f"annotation missing 'positionID': keys={list(entry.keys())}")
            world_pt = wildtrack.position_id_to_world(np.array([entry["positionID"]]))
            # reproject the ground point through this camera's calibration
            from densewalk import frames as fr

            pix = fr.world_to_pixel(world_pt, world_to_cam, intr)[0]
            px, py = int(round(pix[0])), int(round(pix[1]))
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)

            if "views" in entry and view_idx < len(entry["views"]):
                v = entry["views"][view_idx]
                if v.get("xmin", -1) != -1:
                    cv2.rectangle(
                        img,
                        (int(v["xmin"]), int(v["ymin"])),
                        (int(v["xmax"]), int(v["ymax"])),
                        (0, 255, 0),
                        2,
                    )

        out_path = args.out_dir / f"view{view_idx}_{intr_path.stem}_overlay.png"
        cv2.imwrite(str(out_path), img)
        print(f"  saved {out_path}  (green=annotated bbox, red cross=reprojected ground point)")

    print(
        "\nDone. Inspect the overlays: red crosses should land inside/near the "
        "corresponding green boxes (at the person's feet, since positionID is a "
        "ground-plane position). If they don't, the schema/camera-correspondence "
        "assumptions are wrong -- fix before trusting milestone1_visibility_report.py."
    )


if __name__ == "__main__":
    sys.exit(main())
