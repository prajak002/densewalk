"""M1-Varanasi step 1: video -> frames, with a blur/QC report.

Unlike AV2 there is no rig calibration and no lidar here: every pose must
come from SfM, and SfM fails on motion-blurred frames. So this step is not
just an extract -- it scores each frame's sharpness (variance of the
Laplacian) so blurred frames can be dropped before COLMAP rather than
silently poisoning the reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video.name}: {w}x{h} @ {fps:.2f}fps, {n_total} frames")

    report = []
    idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            name = f"{idx:05d}.jpg"
            cv2.imwrite(str(img_dir / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            report.append({
                "name": name,
                "frame": idx,
                "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                "mean_luma": float(grey.mean()),
            })
            written += 1
        idx += 1
    cap.release()

    sharp = np.array([r["sharpness"] for r in report])
    # Frames well below the clip's own median are the blurred ones; an
    # absolute threshold would not transfer between clips.
    thresh = float(np.median(sharp) * 0.6)
    for r in report:
        r["blurred"] = bool(r["sharpness"] < thresh)
    n_blur = sum(r["blurred"] for r in report)

    summary = {
        "video": str(args.video),
        "width": w, "height": h, "fps": fps,
        "frames_decoded": idx, "frames_written": written,
        "sharpness_median": float(np.median(sharp)),
        "sharpness_p10": float(np.percentile(sharp, 10)),
        "blur_threshold": thresh,
        "n_blurred": n_blur,
    }
    (args.out_dir / "frames_report.json").write_text(
        json.dumps({"summary": summary, "frames": report}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"{n_blur}/{written} frames flagged blurred -> {args.out_dir}/frames_report.json")


if __name__ == "__main__":
    main()
