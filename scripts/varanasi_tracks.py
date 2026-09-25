"""M1-Varanasi step 4: the crowd as persistent tracks, not per-frame blobs.

The central problem is humanoid locomotion *among* moving people and
two-wheelers, so the crowd is the payload, not noise. A per-frame mask cannot
support that: the robot has to know that the person ahead is the same person
as last frame and is closing at 1.3 m/s. That requires identity over time.

Emits one record per (track_id, frame) with the box, the class, and the
FOOT POINT -- the bottom-centre of the box, which is where the person meets
the ground. Once COLMAP gives the camera poses and the ground plane, that
foot point back-projects to a metric 3D position, and the sequence of them is
the obstacle trajectory M7/M8 consume.

Foot point rather than box centre is deliberate: the centre floats with the
person's height and pose, the feet are on the plane we can actually solve for.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Same dynamic set as varanasi_masks.py; kept in sync deliberately.
DYNAMIC_COCO_IDS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolo11x-seg.pt")
    ap.add_argument("--tracker", default="botsort.yaml",
                    help="botsort keeps IDs through occlusion better than bytetrack, "
                         "which matters in a crowd this dense")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-track-len", type=int, default=5,
                    help="drop flicker: a 1-2 frame track is a false positive, "
                         "not an obstacle")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg under {args.images}")

    model = YOLO(args.weights)
    per_track = defaultdict(list)
    n_untracked = 0

    # One image per call with persist=True is the documented way to keep the
    # tracker's state across a custom loop. Handing track() the whole 361-path
    # list instead makes ultralytics build one batch of 361 and it OOMs the
    # 16GB card (tried to allocate 7.93 GiB).
    def frames():
        for path in paths:
            yield path, model.track(source=str(path), tracker=args.tracker,
                                    conf=args.conf, persist=True, verbose=False)[0]

    for frame_idx, (path, r) in enumerate(frames()):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        if r.boxes.id is None:
            n_untracked += len(r.boxes)
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        for tid, c, (x0, y0, x1, y1) in zip(ids, cls, xyxy):
            if c not in DYNAMIC_COCO_IDS:
                continue
            per_track[int(tid)].append({
                "frame": frame_idx,
                "image": path.name,
                "cls": DYNAMIC_COCO_IDS[c],
                "xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "foot_px": [float((x0 + x1) / 2), float(y1)],
                "height_px": float(y1 - y0),
            })

    tracks = {k: v for k, v in per_track.items() if len(v) >= args.min_track_len}
    lens = np.array([len(v) for v in tracks.values()]) if tracks else np.array([0])
    by_cls = defaultdict(int)
    for v in tracks.values():
        by_cls[v[0]["cls"]] += 1

    summary = {
        "n_frames": len(paths),
        "n_tracks_raw": len(per_track),
        "n_tracks_kept": len(tracks),
        "min_track_len": args.min_track_len,
        "track_len_mean": float(lens.mean()),
        "track_len_max": int(lens.max()),
        "tracks_by_class": dict(by_cls),
        "detections_without_id": n_untracked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary, "tracks": {str(k): v for k, v in tracks.items()}}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
