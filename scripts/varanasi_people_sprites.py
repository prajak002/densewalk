"""Extract each tracked person/vehicle as an RGBA cutout from the source video.

The splat deliberately has the crowd masked out of it -- that is what makes the
static background clean. But the crowd is the point of this project, so they
have to come back, and they should come back as the REAL people from the video
rather than as capsules or synthetic avatars.

Per detection this writes the image crop with the instance's own segmentation
mask as alpha. Re-running the tracker (rather than reusing the combined
per-frame mask) keeps instances separate: with a merged mask, overlapping
people in a crowd this dense cut each other's silhouettes apart.

Each sprite is tagged with the metric ground position and height it was seen
at, so the renderer can place it as a billboard in the reconstructed street.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DYNAMIC_COCO_IDS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", default="yolo11x-seg.pt")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-track-len", type=int, default=5)
    ap.add_argument("--min-px", type=int, default=18,
                    help="skip cutouts smaller than this; they are mush at 640x360")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = sorted(args.images.glob("*.jpg"))
    model = YOLO(args.weights)
    sprite_dir = args.out_dir / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)

    per_track = defaultdict(list)
    for frame_idx, path in enumerate(paths):
        r = model.track(source=str(path), tracker=args.tracker, conf=args.conf,
                        imgsz=args.imgsz, persist=True, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None or r.masks is None:
            continue
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        data = r.masks.data.cpu().numpy()

        for tid, c, box, m in zip(ids, cls, xyxy, data):
            if c not in DYNAMIC_COCO_IDS:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if (x1 - x0) < args.min_px or (y1 - y0) < args.min_px:
                continue
            mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            alpha = (mask[y0:y1, x0:x1] > 0.5).astype(np.uint8) * 255
            if alpha.sum() == 0:
                continue
            crop = img[y0:y1, x0:x1]
            rgba = np.dstack([crop, alpha])
            out_t = sprite_dir / str(int(tid))
            out_t.mkdir(exist_ok=True)
            cv2.imwrite(str(out_t / f"{frame_idx:05d}.png"), rgba)
            per_track[int(tid)].append({
                "frame": frame_idx, "image": path.name,
                "cls": DYNAMIC_COCO_IDS[c],
                "xyxy": [float(v) for v in box],
                "w_px": int(x1 - x0), "h_px": int(y1 - y0),
                "file": f"{int(tid)}/{frame_idx:05d}.png",
            })

        if frame_idx % 60 == 0:
            print(f"  {frame_idx}/{len(paths)}", flush=True)

    tracks = {str(k): v for k, v in per_track.items() if len(v) >= args.min_track_len}
    n_sprites = sum(len(v) for v in tracks.values())
    summary = {
        "n_tracks": len(tracks), "n_sprites": n_sprites,
        "by_class": {c: sum(1 for v in tracks.values() if v[0]["cls"] == c)
                     for c in sorted({v[0]["cls"] for v in tracks.values()})},
    }
    (args.out_dir / "sprites.json").write_text(
        json.dumps({"summary": summary, "tracks": tracks}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
