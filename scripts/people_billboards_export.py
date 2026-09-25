"""Pack the real-pixel people cutouts for the three.js viewer (outputs/scene3d/people/).

Each person is shown exactly as the video shows them: per video frame, the RGBA
cutout of that person (varanasi_people_sprites.py, instance mask as alpha),
standing at their metric ground position (crowd_world track, via the
sprite-to-metric match from varanasi_match_sprites.py).

Size in metres comes from the pinhole relation h_m = h_px * depth / f, with depth
the distance from the phone camera to the person's foot point along the camera's
forward axis at that frame (camera poses and f from the same COLMAP model).

Output per metric track: one WebP atlas (frames in a grid, fixed cell) plus an
index with, per frame, the cell, the metric width/height and the ground xy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=Path, required=True, help="dir with sprites.json, match.json, sprites/")
    ap.add_argument("--crowd", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True, help="outputs/scene3d/scene.json (camera path)")
    ap.add_argument("--focal-px", type=float, default=445.1)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--cell-max", type=int, default=160, help="atlas cell height cap (px)")
    ap.add_argument("--classes", default="person")
    ap.add_argument("--sr-frames", type=Path, default=None, help="x4 super-resolved frames: crop people from these")
    ap.add_argument("--sr-scale", type=int, default=4)
    ap.add_argument("--atlas-w", type=int, default=2048)
    ap.add_argument("--atlas-h-max", type=int, default=8192)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sprites = json.loads((args.people / "sprites.json").read_text())["tracks"]
    match = json.loads((args.people / "match.json").read_text())
    crowd = json.loads(args.crowd.read_text())["tracks"]
    S = json.loads(args.scene.read_text())
    cam, fwd = np.array(S["camera"]["xyz"]), np.array(S["camera"]["fwd"])
    keep_cls = set(args.classes.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    index, n_frames_total, depths = {}, 0, []
    sr_cache = {}
    for mtid, m in match.items():
        if mtid not in crowd or crowd[mtid]["cls"] not in keep_cls:
            continue
        tr = crowd[mtid]
        tt, xy = np.array(tr["t_s"]), np.array(tr["xy_m"])
        dets = sorted(sprites[m["sprite_id"]], key=lambda d: d["frame"])
        rows = []
        for d in dets:
            f = d["frame"]
            ts = f / args.fps
            if ts < tt[0] - 0.1 or ts > tt[-1] + 0.1:
                continue                                    # outside the metric track: no position
            p = np.array([np.interp(ts, tt, xy[:, 0]), np.interp(ts, tt, xy[:, 1]), 0.0])
            ci = min(f, len(cam) - 1)
            depth = float(np.dot(p - cam[ci], fwd[ci] / np.linalg.norm(fwd[ci])))
            if depth < 0.5:
                continue
            img = cv2.imread(str(args.people / "sprites" / d["file"]), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if args.sr_frames is not None:
                # same person, same box, from the x4 frame; the instance mask is upsampled with soft edges
                k = args.sr_scale
                fr_sr = sr_cache.get(d["image"])
                if fr_sr is None:
                    fr_sr = cv2.imread(str(args.sr_frames / d["image"]))
                    sr_cache.clear(); sr_cache[d["image"]] = fr_sr
                x0, y0 = max(0, int(round(d["xyxy"][0]))), max(0, int(round(d["xyxy"][1])))
                h0, w0 = img.shape[:2]
                crop = fr_sr[y0 * k:(y0 + h0) * k, x0 * k:(x0 + w0) * k]
                a = cv2.resize(img[:, :, 3], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
                a = cv2.GaussianBlur(a, (0, 0), k * 0.6)
                a = np.clip((a.astype(np.float32) - 64) * (255 / 128), 0, 255).astype(np.uint8)
                img = np.dstack([crop, a])
            h_m = d["h_px"] * depth / args.focal_px
            w_m = d["w_px"] * depth / args.focal_px
            rows.append((f, img, w_m, h_m, p[:2]))
            depths.append(depth)
        if len(rows) < 3:
            continue
        ch = min(args.cell_max, max(r[1].shape[0] for r in rows))
        aspect = max(r[1].shape[1] / r[1].shape[0] for r in rows)
        while True:                       # largest cell height whose grid fits one atlas page
            cw = max(1, int(math.ceil(aspect * ch)))
            cols = max(1, min(len(rows), args.atlas_w // cw))
            nrow = math.ceil(len(rows) / cols)
            if nrow * ch <= args.atlas_h_max or ch <= 32:
                break
            ch = int(ch * 0.9)
        atlas = np.zeros((nrow * ch, cols * cw, 4), np.uint8)
        frames = []
        for k, (f, img, w_m, h_m, pxy) in enumerate(rows):
            s = ch / img.shape[0]
            im = cv2.resize(img, (min(cw, max(1, int(round(img.shape[1] * s)))), ch), interpolation=cv2.INTER_AREA)
            r0, c0 = (k // cols) * ch, (k % cols) * cw
            atlas[r0:r0 + ch, c0:c0 + im.shape[1]] = im
            frames.append({"f": f, "u": c0, "v": r0, "w": im.shape[1], "h": ch,
                           "wm": round(w_m, 3), "hm": round(h_m, 3), "xy": [round(float(pxy[0]), 3), round(float(pxy[1]), 3)]})
        name = f"p{mtid}.webp"
        cv2.imwrite(str(args.out / name), atlas, [cv2.IMWRITE_WEBP_QUALITY, 90])
        index[mtid] = {"cls": tr["cls"], "atlas": name, "size": [atlas.shape[1], atlas.shape[0]], "frames": frames}
        n_frames_total += len(frames)

    hs = [fr["hm"] for v in index.values() for fr in v["frames"] if fr["h"] >= 60]
    summary = {"tracks": len(index), "sprite_frames": n_frames_total,
               "depth_m_p5_p50_p95": np.percentile(depths, [5, 50, 95]).round(2).tolist(),
               "height_m_p10_p50_p90_(boxes>=60px)": np.percentile(hs, [10, 50, 90]).round(2).tolist() if hs else None,
               "note": "height from box height incl. occluded/cropped boxes; p50 should be ~1.6-1.7 m"}
    (args.out / "people.json").write_text(json.dumps({"summary": summary, "tracks": index}, separators=(",", ":")))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
