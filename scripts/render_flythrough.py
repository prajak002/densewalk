"""Step 6: render a smooth fly-through of the trained background splat.

Camera path (all in the AV2 city frame, metres, Z-up):
  1. start at one end of the reconstructed drive
  2. fly forward along the street
  3. slow arc around a point of interest
  4. pull back and up to a wide overhead view
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

from densewalk import frames
from densewalk.av2 import Av2Sequence


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])) -> frames.Pose:
    """World->camera pose with OpenCV axes (X right, Y down, Z forward)."""
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    fwd = fwd / n
    if abs(float(np.dot(fwd, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_cam_from_world = np.stack([right, down, fwd], axis=0)
    return frames.Pose(R_cam_from_world, -R_cam_from_world @ eye)


def smoothstep(t):
    return t * t * (3 - 2 * t)


def catmull_rom(points: np.ndarray, n: int) -> np.ndarray:
    """Smooth interpolation through control points."""
    p = np.asarray(points, dtype=np.float64)
    p = np.vstack([p[0], p, p[-1]])
    segs = len(p) - 3
    out = []
    for i in range(n):
        u = i / max(1, n - 1) * segs
        k = min(int(u), segs - 1)
        t = u - k
        p0, p1, p2, p3 = p[k], p[k + 1], p[k + 2], p[k + 3]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                          (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.stack(out)


def build_path(seq, n_frames: int):
    """Compose the four-phase path from the real ego trajectory."""
    ts = seq.image_timestamps("ring_front_center")
    ego = np.stack([seq.city_se3_ego(t).t for t in ts])
    start, end = ego[0], ego[-1]
    heading = end - start
    heading[2] = 0
    heading /= np.linalg.norm(heading)
    left = np.cross(np.array([0, 0, 1.0]), heading)
    mid = ego[len(ego) // 2]
    centre = ego.mean(axis=0)
    road_z = float(np.median(ego[:, 2]))

    n1 = int(n_frames * 0.42)   # fly forward
    n2 = int(n_frames * 0.30)   # arc around point of interest
    n3 = n_frames - n1 - n2     # pull back / up

    eyes, targets = [], []

    # phase 1: forward down the street, slightly above the ego path
    ctrl = ego[:: max(1, len(ego) // 12)] + np.array([0, 0, 2.2])
    p1 = catmull_rom(ctrl, n1)
    for i, e in enumerate(p1):
        t = i / max(1, n1 - 1)
        ahead = min(len(ego) - 1, int(t * (len(ego) - 1)) + 25)
        eyes.append(e)
        targets.append(ego[ahead] + np.array([0, 0, 1.0]))

    # phase 2: slow arc around the mid-point of interest
    poi = mid + np.array([0, 0, 1.2])
    r = 11.0
    a0 = np.arctan2(p1[-1][1] - poi[1], p1[-1][0] - poi[0])
    for i in range(n2):
        t = smoothstep(i / max(1, n2 - 1))
        a = a0 + t * np.deg2rad(150)
        h = 2.5 + 2.5 * t
        eyes.append(poi + np.array([r * np.cos(a), r * np.sin(a), h]))
        targets.append(poi)

    # phase 3: pull back and up to a wide overhead
    e0 = eyes[-1]
    e1 = centre - heading * 26.0 + np.array([0, 0, 17.0])
    for i in range(n3):
        t = smoothstep(i / max(1, n3 - 1))
        eyes.append(e0 * (1 - t) + e1 * t)
        targets.append(poi * (1 - t) + (centre + np.array([0, 0, road_z * 0])) * t)

    return np.stack(eyes), np.stack(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("outputs/background_model.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/m1_background.mp4"))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fov-deg", type=float, default=52.0)  # match the ~47 deg training cameras
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.model, map_location=device)
    sh_degree = int(ck["sh_degree"])
    means = ck["means"].to(device)
    quats = ck["quats"].to(device)
    scales = torch.exp(ck["scales"].to(device))
    opacities = torch.sigmoid(ck["opacities"].to(device))
    colors = torch.cat([ck["sh0"].to(device), ck["shN"].to(device)], dim=1)
    print(f"loaded {means.shape[0]} gaussians, metrics={ck.get('metrics')}")

    seq = Av2Sequence(args.seq_dir)
    eyes, targets = build_path(seq, args.frames)
    # the model was trained in a recentred world -- move the path with it
    origin_path = Path(str(args.model) + ".origin.pt")
    if origin_path.exists():
        origin = torch.load(origin_path)["origin"].numpy()
        eyes = eyes - origin
        targets = targets - origin
        print(f"applied training recentre {origin.round(1)}")

    W, H = args.width, args.height
    f = 0.5 * W / np.tan(np.deg2rad(args.fov_deg) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]

    tmp = args.out.parent / "_flythrough_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()

    import cv2
    with torch.no_grad():
        for i, (eye, tgt) in enumerate(zip(eyes, targets)):
            pose = look_at(eye, tgt)
            viewmat = torch.tensor(pose.as_matrix(), dtype=torch.float32, device=device)[None]
            img, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat, Ks=K, width=W, height=H,
                sh_degree=sh_degree, camera_model="pinhole",
                with_ut=True, with_eval3d=True, packed=False,
                near_plane=0.2, far_plane=400.0,
            )
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(tmp / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % 60 == 0:
                print(f"  frame {i}/{len(eyes)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.out)]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
