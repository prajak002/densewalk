"""Map sprite track ids to metric track ids.

The sprites came from a second tracking pass, and a tracker assigns its own ids
each run -- so sprite track 7 and metric track 7 are different people. Match
them by box overlap on the frames they share, which is unambiguous because both
passes saw the same pixels.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--sprites", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-iou", type=float, default=0.35)
    args = ap.parse_args()

    metric = json.loads(args.tracks.read_text())["tracks"]
    sprites = json.loads(args.sprites.read_text())["tracks"]

    by_frame = defaultdict(list)
    for stid, dets in sprites.items():
        for d in dets:
            by_frame[d["frame"]].append((stid, d["xyxy"]))

    scores = defaultdict(list)
    for mtid, dets in metric.items():
        for d in dets:
            for stid, sbox in by_frame.get(d["frame"], []):
                v = iou(d["xyxy"], sbox)
                if v > 0:
                    scores[(mtid, stid)].append(v)

    best = {}
    for (mtid, stid), vals in scores.items():
        mean_iou = float(np.mean(vals))
        n = len(vals)
        if mean_iou < args.min_iou:
            continue
        cur = best.get(mtid)
        if cur is None or (mean_iou * n) > (cur["mean_iou"] * cur["n"]):
            best[mtid] = {"sprite_id": stid, "mean_iou": mean_iou, "n": n}

    args.out.write_text(json.dumps(best, indent=2))
    print(json.dumps({
        "metric_tracks": len(metric), "sprite_tracks": len(sprites),
        "matched": len(best),
        "mean_iou": round(float(np.mean([b["mean_iou"] for b in best.values()])), 3)
        if best else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
