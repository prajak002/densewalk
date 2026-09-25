"""Step 6 (chunked): render the fly-through from several per-segment
models plus one coarse global model.

A single model cannot hold 83m of street at high density on a 12GB card.
So the drive is split into overlapping chunks, each trained densely, and
the camera path is rendered with whichever chunk contains the camera --
cross-fading where chunks overlap. The wide overhead phase uses the
coarse global model, where fine detail is invisible anyway (level of
detail).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

from densewalk import frames
from densewalk.av2 import Av2Sequence
from render_flythrough import build_path, look_at


def load_model(path: Path, device):
    ck = torch.load(path, map_location=device)
    origin_p = Path(str(path) + ".origin.pt")
    origin = torch.load(origin_p)["origin"].numpy() if origin_p.exists() else np.zeros(3)
    return {
        "means": ck["means"].to(device),
        "quats": ck["quats"].to(device),
        "scales": torch.exp(ck["scales"].to(device)),
        "opacities": torch.sigmoid(ck["opacities"].to(device)),
        "colors": torch.cat([ck["sh0"].to(device), ck["shN"].to(device)], dim=1),
        "sh_degree": int(ck.get("sh_degree", 3)),
        "origin": origin,
        "n": int(ck["means"].shape[0]),
        "path": str(path),
    }


def render_with(model, eye, tgt, K, W, H, device):
    pose = look_at(eye - model["origin"], tgt - model["origin"])
    viewmat = torch.tensor(pose.as_matrix(), dtype=torch.float32, device=device)[None]
    img, _, _ = rasterization(
        means=model["means"], quats=model["quats"], scales=model["scales"],
        opacities=model["opacities"], colors=model["colors"],
        viewmats=viewmat, Ks=K, width=W, height=H,
        sh_degree=model["sh_degree"], camera_model="pinhole",
        with_ut=True, with_eval3d=True, packed=False,
        near_plane=0.2, far_plane=400.0,
    )
    return img[0].clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True,
                    help="per-segment models, in order along the drive")
    ap.add_argument("--global-model", type=Path, default=None,
                    help="coarse model used for the wide overhead phase")
    ap.add_argument("--out", type=Path, default=Path("outputs/m1_background.mp4"))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fov-deg", type=float, default=52.0)
    ap.add_argument("--blend-m", type=float, default=6.0)
    args = ap.parse_args()

    device = "cuda"
    seq = Av2Sequence(args.seq_dir)
    eyes, targets = build_path(seq, args.frames)

    # arc-length of the ego drive, used to assign chunks to path positions
    ego_ts = seq.image_timestamps("ring_front_center")
    ego = np.stack([seq.city_se3_ego(t).t for t in ego_ts])
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ego, axis=0), axis=1))])
    total = float(dist[-1])
    n_chunks = len(args.chunks)
    bounds = [(i * total / n_chunks, (i + 1) * total / n_chunks) for i in range(n_chunks)]
    print(f"drive {total:.1f}m split into {n_chunks} chunks: "
          + ", ".join(f"[{a:.0f},{b:.0f}]m" for a, b in bounds))

    W, H = args.width, args.height
    f = 0.5 * W / np.tan(np.deg2rad(args.fov_deg) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]

    tmp = args.out.parent / "_flythrough_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()

    def along_for(eye):
        return float(dist[int(np.argmin(np.linalg.norm(ego - eye, axis=1)))])

    def height_above_road(eye):
        return float(eye[2] - np.median(ego[:, 2]))

    # render chunk by chunk so only one or two models sit in VRAM at once
    assign = []
    for i, eye in enumerate(eyes):
        if args.global_model is not None and height_above_road(eye) > 9.0:
            assign.append(("global", None))
            continue
        a = along_for(eye)
        k = min(n_chunks - 1, max(0, int(a / (total / n_chunks))))
        lo, hi = bounds[k]
        # blend weight toward the neighbouring chunk near a boundary
        nb, w = None, 0.0
        if a > hi - args.blend_m and k + 1 < n_chunks:
            nb, w = k + 1, (a - (hi - args.blend_m)) / args.blend_m
        elif a < lo + args.blend_m and k - 1 >= 0:
            nb, w = k - 1, ((lo + args.blend_m) - a) / args.blend_m
        assign.append((k, (nb, float(np.clip(w, 0, 1))) if nb is not None else None))

    need = sorted({a for a, _ in assign if a != "global"} |
                  {b[0] for _, b in assign if b is not None})
    print("frames per chunk:", {k: sum(1 for a, _ in assign if a == k) for k in need},
          "global:", sum(1 for a, _ in assign if a == "global"))

    rendered = {}
    for k in need:
        model = load_model(args.chunks[k], device)
        print(f"chunk {k}: {model['n']} gaussians from {model['path']}")
        for i, (a, nb) in enumerate(assign):
            if a != k and (nb is None or nb[0] != k):
                continue
            img = render_with(model, eyes[i], targets[i], K, W, H, device)
            rendered.setdefault(i, {})[k] = img.cpu()
        del model
        torch.cuda.empty_cache()

    if args.global_model is not None and any(a == "global" for a, _ in assign):
        gm = load_model(args.global_model, device)
        print(f"global: {gm['n']} gaussians (wide overhead phase)")
        for i, (a, _) in enumerate(assign):
            if a == "global":
                rendered.setdefault(i, {})["global"] = render_with(
                    gm, eyes[i], targets[i], K, W, H, device).cpu()
        del gm
        torch.cuda.empty_cache()

    for i in range(len(eyes)):
        a, nb = assign[i]
        parts = rendered[i]
        if a == "global":
            img = parts["global"]
        elif nb is not None and nb[0] in parts:
            img = parts[a] * (1 - nb[1]) + parts[nb[0]] * nb[1]
        else:
            img = parts[a]
        frame = (img.numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.out)],
                   check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
