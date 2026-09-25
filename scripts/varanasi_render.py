"""M1-Varanasi step 6: render a fly-through from the trained splat.

Renders along the solved camera path, which is the only trajectory this clip
can support honestly: every surface was seen from one narrow range of angles,
so a viewpoint far off the path shows the parts of the scene that were never
observed. --smooth fits a low-order polynomial through the camera centres to
take the walking bob out without inventing new viewpoints.

Output is the Varanasi equivalent of outputs/m1_background.mp4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames
from densewalk.video import H264Writer
from varanasi_train_splat import read_colmap


def smooth_path(C: np.ndarray, order: int = 5) -> np.ndarray:
    """Fit a low-order polynomial per axis against normalised arclength."""
    t = np.linspace(0.0, 1.0, len(C))
    out = np.empty_like(C)
    for a in range(3):
        out[:, a] = np.polyval(np.polyfit(t, C[:, a], order), t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--smooth", type=int, default=5,
                    help="polynomial order for the camera path; 0 = raw poses")
    ap.add_argument("--scale-res", type=int, default=2,
                    help="render at NxN the source resolution")
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device)
    origin = torch.load(str(args.ckpt) + ".origin.pt")["origin"].numpy().astype(np.float64)
    sh_degree = int(ck["sh_degree"])
    print(f"checkpoint: {ck['means'].shape[0]} gaussians, sh_degree {sh_degree}")
    if "metrics" in ck:
        print("metrics:", ck["metrics"])

    params = {k: ck[k].to(device) for k in
              ("means", "scales", "quats", "opacities", "sh0", "shN")}

    cam, imgs, _, _ = read_colmap(args.model_dir)
    W = cam["w"] * args.scale_res
    H = cam["h"] * args.scale_res
    K = torch.tensor([[cam["f"] * args.scale_res, 0, cam["cx"] * args.scale_res],
                      [0, cam["f"] * args.scale_res, cam["cy"] * args.scale_res],
                      [0, 0, 1]], dtype=torch.float32, device=device)[None]
    radial = torch.tensor([cam["k1"], 0, 0, 0, 0, 0],
                          dtype=torch.float32, device=device)[None]

    # scene was recentred at training time; move the cameras the same way
    shift = frames.Pose(np.eye(3), -origin)
    poses = [im["w2c"].compose(shift.inverse()) for im in imgs]

    if args.smooth > 0:
        C = np.stack([p.inverse().apply(np.zeros((1, 3)))[0] for p in poses])
        Cs = smooth_path(C, args.smooth)
        poses = [frames.Pose(p.R, -p.R @ c) for p, c in zip(poses, Cs)]
        print(f"smoothed camera path (order {args.smooth}), "
              f"max shift {np.abs(Cs - C).max():.3f} units")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = H264Writer(args.out, W, H, args.fps)

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    with torch.no_grad():
        for i, pose in enumerate(poses):
            vm = torch.tensor(pose.as_matrix(), dtype=torch.float32, device=device)[None]
            img, _, _ = rasterization(
                means=params["means"], quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors, viewmats=vm, Ks=K, width=W, height=H,
                sh_degree=sh_degree, camera_model="pinhole",
                with_ut=True, with_eval3d=True, packed=False,
                radial_coeffs=radial, near_plane=0.01, far_plane=100.0)
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % 60 == 0:
                print(f"  {i}/{len(poses)}", flush=True)
    vw.release()
    print(f"wrote {args.out} ({len(poses)} frames at {W}x{H})")


if __name__ == "__main__":
    main()
