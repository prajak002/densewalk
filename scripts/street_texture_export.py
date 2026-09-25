"""Photo-textured street model from the video: clean from any viewpoint, crowd removed.

Geometry (metric, Z-up, floor z=0): two vertical facade planes along the walkway edges
and the ground plane between them. Plane positions come from the COLMAP points
(right shopfronts ~ -3.2 m, left stall line ~ +3 m; see the per-slab histogram).

Texture: for every texel, the colour of the x4 super-resolved frame that saw that point
from closest and most head-on, skipping pixels the dynamic mask (masks_train, 255 =
person/vehicle) marks as moving -- so the crowd is erased from walls and floor and only
the static market remains. Projection uses the COLMAP camera (SIMPLE_RADIAL, k1) and
world.json, every coordinate step through frames.Pose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from densewalk import frames  # noqa: E402
from varanasi_train_splat import read_colmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--sr-frames", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="masks_train/<name>.png, 255 = dynamic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--x0", type=float, default=-16.0); ap.add_argument("--x1", type=float, default=32.0)
    ap.add_argument("--y-left", type=float, default=3.4); ap.add_argument("--y-right", type=float, default=-3.6)
    ap.add_argument("--wall-h", type=float, default=9.0)
    ap.add_argument("--res-m", type=float, default=0.015)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--sr", type=int, default=4)
    a = ap.parse_args()

    w = json.loads(a.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))
    to_colmap = canon.inverse()
    cam, imgs, _, _ = read_colmap(a.model_dir)
    f, cx, cy, k1 = cam["f"] * a.sr, cam["cx"] * a.sr, cam["cy"] * a.sr, cam["k1"]
    Wimg, Himg = cam["w"] * a.sr, cam["h"] * a.sr

    L = a.x1 - a.x0
    nx = int(L / a.res_m)
    planes = {
        # name: (texel -> world point grid), normal
        "wall_left": dict(nu=nx, nv=int(a.wall_h / a.res_m), normal=np.array([0, -1.0, 0]),
                          pt=lambda u, v: np.stack([a.x0 + u * a.res_m, np.full_like(u, a.y_left), a.wall_h - v * a.res_m], -1)),
        "wall_right": dict(nu=nx, nv=int(a.wall_h / a.res_m), normal=np.array([0, 1.0, 0]),
                           pt=lambda u, v: np.stack([a.x0 + u * a.res_m, np.full_like(u, a.y_right), a.wall_h - v * a.res_m], -1)),
        "ground": dict(nu=nx, nv=int((a.y_left - a.y_right) / a.res_m), normal=np.array([0, 0, 1.0]),
                       pt=lambda u, v: np.stack([a.x0 + u * a.res_m, a.y_left - v * a.res_m, np.zeros_like(u)], -1)),
    }
    for p in planes.values():
        uu, vv = np.meshgrid(np.arange(p["nu"]) + 0.5, np.arange(p["nv"]) + 0.5)
        p["X"] = p["pt"](uu, vv).reshape(-1, 3)                       # metric world
        p["Xc"] = to_colmap.apply(p["X"] / scale)                     # COLMAP world
        p["best"] = np.full(len(p["X"]), np.inf, np.float32)
        p["rgb"] = np.zeros((len(p["X"]), 3), np.uint8)

    used = 0
    for im in imgs[:: a.frame_step]:
        img = cv2.imread(str(a.sr_frames / im["name"]))
        m = cv2.imread(str(a.masks / f"{im['name']}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        used += 1
        w2c = im["w2c"]
        C_metric = canon.apply(w2c.inverse().t[None])[0] * scale
        for p in planes.values():
            pc = w2c.apply(p["Xc"])
            z = pc[:, 2]
            front = z > 0.05
            xn, yn = pc[:, 0] / np.where(front, z, 1), pc[:, 1] / np.where(front, z, 1)
            r2 = xn * xn + yn * yn
            u = f * xn * (1 + k1 * r2) + cx
            v = f * yn * (1 + k1 * r2) + cy
            ok = front & (u >= 2) & (v >= 2) & (u < Wimg - 2) & (v < Himg - 2)
            idx = np.nonzero(ok)[0]
            ui, vi = u[idx].astype(int), v[idx].astype(int)
            static = m[vi // a.sr, ui // a.sr] < 128
            idx, ui, vi = idx[static], ui[static], vi[static]
            d = p["X"][idx] - C_metric
            dist = np.linalg.norm(d, axis=1)
            cosang = np.abs(d @ p["normal"]) / np.maximum(dist, 1e-6)
            score = (dist / np.maximum(cosang, 0.15)).astype(np.float32)
            better = score < p["best"][idx]
            idx, ui, vi = idx[better], ui[better], vi[better]
            p["best"][idx] = score[better]
            p["rgb"][idx] = img[vi, ui]
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"x0": a.x0, "x1": a.x1, "y_left": a.y_left, "y_right": a.y_right, "wall_h": a.wall_h,
            "res_m": a.res_m, "frames_used": used, "planes": {}}
    for name, p in planes.items():
        tex = p["rgb"].reshape(p["nv"], p["nu"], 3)
        seen = np.isfinite(p["best"]).reshape(p["nv"], p["nu"])
        if (~seen).any() and seen.any():           # unseen texels: smooth fill from surrounding seen colour
            k = 8
            small = cv2.resize(tex, (tex.shape[1] // k, tex.shape[0] // k), interpolation=cv2.INTER_AREA)
            smask = cv2.resize((~seen).astype(np.uint8) * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            fill = cv2.resize(cv2.inpaint(small, smask, 5, cv2.INPAINT_TELEA), (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_CUBIC)
            tex = np.where(seen[..., None], tex, fill)
        cv2.imwrite(str(a.out / f"{name}.jpg"), tex, [cv2.IMWRITE_JPEG_QUALITY, 88])
        meta["planes"][name] = {"size_px": [p["nu"], p["nv"]], "coverage": round(float(seen.mean()), 3)}
    (a.out / "street.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
