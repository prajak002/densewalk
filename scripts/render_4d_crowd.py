"""M2: render the time-varying scene — static background + moving people.

scene(t) = background gaussians (fixed) + per-person gaussians posed at t

Both are concatenated into ONE gsplat rasterization call per frame, so
people are depth-sorted against the background correctly rather than
being composited as a flat overlay.

The camera follows the original ego trajectory, so the output shows the
street from the driven viewpoint with the crowd actually moving — unlike
the static background-only fly-through, this video is not frozen.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

from densewalk import frames
from densewalk.av2 import Av2Sequence
from densewalk.dynamic_crowd import build_person_clouds, crowd_gaussians_at


def load_chunk(path: Path, device):
    ck = torch.load(path, map_location=device)
    origin_p = Path(str(path) + ".origin.pt")
    origin = torch.load(origin_p)["origin"].numpy() if origin_p.exists() else np.zeros(3)
    return {
        "means": ck["means"].to(device),
        "quats": ck["quats"].to(device),
        "scales": torch.exp(ck["scales"].to(device)),
        "opacities": torch.sigmoid(ck["opacities"].to(device)),
        "sh0": ck["sh0"].to(device),
        "origin": origin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/m2_4d_crowd.mp4"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--max-people", type=int, default=40)
    ap.add_argument("--points-per-person", type=int, default=900)
    ap.add_argument("--cam-height", type=float, default=4.5)
    ap.add_argument("--look-ahead", type=int, default=45)
    args = ap.parse_args()

    device = "cuda"
    seq = Av2Sequence(args.seq_dir)

    print("building person clouds from ground-truth tracks ...", flush=True)
    clouds = build_person_clouds(
        seq,
        cameras=["ring_front_center", "ring_front_left", "ring_front_right"],
        points_per_person=args.points_per_person,
        max_people=args.max_people,
    )
    print(f"  {len(clouds)} moving identities "
          f"({sum(c.means_local.shape[0] for c in clouds)} gaussians)", flush=True)
    if not clouds:
        raise RuntimeError("no moving tracks found — nothing dynamic to render")

    # one background chunk at a time would mean reloading per frame; the
    # street fits in 96GB so load them all and concatenate once.
    bg_means, bg_quats, bg_scales, bg_opac, bg_rgb = [], [], [], [], []
    origin = None
    for p in args.chunks:
        c = load_chunk(p, device)
        if origin is None:
            origin = c["origin"]
        shift = torch.tensor(c["origin"] - origin, dtype=torch.float32, device=device)
        bg_means.append(c["means"] + shift)
        bg_quats.append(c["quats"])
        bg_scales.append(c["scales"])
        bg_opac.append(c["opacities"])
        # DC term of the SH -> plain rgb, so background and people share one
        # colour convention in the combined pass
        bg_rgb.append((c["sh0"][:, 0, :] * 0.2820948 + 0.5).clamp(0, 1))
        del c
    bg = {
        "means": torch.cat(bg_means), "quats": torch.cat(bg_quats),
        "scales": torch.cat(bg_scales), "opacities": torch.cat(bg_opac),
        "colors": torch.cat(bg_rgb),
    }
    del bg_means, bg_quats, bg_scales, bg_opac, bg_rgb
    torch.cuda.empty_cache()
    print(f"  background: {bg['means'].shape[0]} gaussians", flush=True)

    cam = seq.camera("ring_front_center")
    stamps = seq.image_timestamps("ring_front_center")
    idx = np.linspace(0, len(stamps) - 1, args.frames).astype(int)

    W, H = args.width, args.height
    f = 0.5 * W / np.tan(np.deg2rad(args.fov_deg) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]

    tmp = args.out.parent / "_m2_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()

    n_people_seen = []
    last_fwd = np.array([1.0, 0.0, 0.0])
    for n, i in enumerate(idx):
        ts = stamps[i]
        ego = seq.city_se3_ego(ts)
        # camera slightly above the ego, looking down the street
        eye = ego.t + np.array([0.0, 0.0, args.cam_height])
        ahead = seq.city_se3_ego(stamps[min(len(stamps) - 1, i + args.look_ahead)]).t
        target = ahead + np.array([0.0, 0.0, 1.0])
        fwd = target - eye
        nrm = np.linalg.norm(fwd)
        # near the end of the drive the look-ahead clamps to the last frame,
        # so target collapses onto eye and fwd degenerates; fall back to the
        # previous heading instead of dividing by ~0.
        if nrm < 1e-3:
            fwd = last_fwd.copy()
        else:
            fwd = fwd / nrm
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fwd, up))) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            fwd = last_fwd.copy()
            right = np.cross(fwd, up)
            rn = np.linalg.norm(right)
        right /= rn
        down = np.cross(fwd, right)
        down /= np.linalg.norm(down)
        fwd = np.cross(right, down)          # re-orthonormalise
        fwd /= np.linalg.norm(fwd)
        last_fwd = fwd
        R = np.stack([right, down, fwd], axis=0)
        w2c = frames.Pose(R, -R @ (eye - origin))

        crowd = crowd_gaussians_at(seq, clouds, ts, device=device, origin=origin)
        if crowd is None:
            means, quats, scales, opac, cols = (
                bg["means"], bg["quats"], bg["scales"], bg["opacities"], bg["colors"])
            n_people_seen.append(0)
        else:
            means = torch.cat([bg["means"], crowd["means"]])
            quats = torch.cat([bg["quats"], crowd["quats"]])
            scales = torch.cat([bg["scales"], crowd["scales"]])
            opac = torch.cat([bg["opacities"], crowd["opacities"]])
            cols = torch.cat([bg["colors"], crowd["colors"]])
            n_people_seen.append(int(crowd["means"].shape[0]))

        viewmat = torch.tensor(w2c.as_matrix(), dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            img, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac,
                colors=cols, viewmats=viewmat, Ks=K, width=W, height=H,
                camera_model="pinhole", with_ut=True, with_eval3d=True,
                packed=False, near_plane=1.5, far_plane=400.0,
            )
        frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(tmp / f"{n:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if n % 50 == 0:
            print(f"  frame {n}/{len(idx)} crowd_gaussians={n_people_seen[-1]}", flush=True)

    arr = np.array(n_people_seen)
    print(f"[M2] crowd gaussians per frame: min={arr.min()} mean={arr.mean():.0f} max={arr.max()}")
    assert arr.max() > 0, "no people were ever rendered — the scene is static"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(args.out)], check=True)
    print(f"[M2] wrote {args.out}")


if __name__ == "__main__":
    main()
