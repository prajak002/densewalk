"""M4 first-person: render the G1's walk through the reconstructed street.

The chase video shows the robot on Isaac's checkerboard, because Isaac holds
the physics and the splat holds the appearance. This renders the SAME walk
from the robot's own eye, inside the Varanasi splat -- which is what "the
robot navigates this street" actually looks like.

The trajectory comes straight from the M4 run, so the pose at frame k is where
the G1 really was at that instant. Chain of frames, all via frames.py:

    sim (M4)  --rotate back-->  canonical metric  --/scale, canon^-1-->  COLMAP

Splats are visual only; nothing here feeds physics.
"""
from __future__ import annotations

import argparse
import json
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

EYE_HEIGHT_M = 1.30     # G1 head height above the floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--crowd", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="sim steps per rendered frame")
    ap.add_argument("--scale-res", type=int, default=2)
    args = ap.parse_args()

    device = "cuda"
    w = json.loads(args.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    run = json.loads(args.run.read_text())
    log = run["log"]
    start_xy = np.array(run["summary"]["start_xy"])
    yaw0 = float(run["summary"]["spawn_yaw"])
    # M4 rotated the world by W2S = R(yaw0); undo it to get back to canonical
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    S2W = np.array([[c0, s0], [-s0, c0]])

    ck = torch.load(args.ckpt, map_location=device)
    origin = torch.load(str(args.ckpt) + ".origin.pt")["origin"].numpy().astype(np.float64)
    sh_degree = int(ck["sh_degree"])
    params = {k: ck[k].to(device) for k in
              ("means", "scales", "quats", "opacities", "sh0", "shN")}
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    print(f"{params['means'].shape[0]} gaussians")

    cam, _, _, _ = read_colmap(args.model_dir)
    W = cam["w"] * args.scale_res
    H = cam["h"] * args.scale_res
    K = torch.tensor([[cam["f"] * args.scale_res, 0, cam["cx"] * args.scale_res],
                      [0, cam["f"] * args.scale_res, cam["cy"] * args.scale_res],
                      [0, 0, 1]], dtype=torch.float32, device=device)[None]
    radial = torch.tensor([cam["k1"], 0, 0, 0, 0, 0],
                          dtype=torch.float32, device=device)[None]
    shift = frames.Pose(np.eye(3), -origin)     # the training-time recentring

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = H264Writer(args.out, W, H, args.fps)

    n = 0
    with torch.no_grad():
        for i in range(0, len(log), args.stride):
            r = log[i]
            # sim -> canonical metric
            xy = S2W @ np.array([r["x"], r["y"]]) + start_xy
            yaw_canon = r["yaw"] + yaw0
            eye_canon = np.array([xy[0], xy[1], EYE_HEIGHT_M])

            # canonical metric -> COLMAP world, then apply the training shift
            p_colmap = canon.inverse().apply((eye_canon / scale)[None])[0]
            p_shift = p_colmap - origin

            # Camera looks along +x of the canonical frame rotated by yaw, with
            # canonical +z up. Build it in canonical space, then rotate into
            # COLMAP space with the same inverse rotation.
            fwd_c = np.array([np.cos(yaw_canon), np.sin(yaw_canon), 0.0])
            up_c = np.array([0.0, 0.0, 1.0])
            right_c = np.cross(fwd_c, up_c); right_c /= np.linalg.norm(right_c)
            down_c = np.cross(fwd_c, right_c)
            # OpenCV camera axes: x right, y down, z forward
            R_c2w_canon = np.stack([right_c, down_c, fwd_c], axis=1)
            Rw = canon.R.T @ R_c2w_canon          # into COLMAP orientation
            R_w2c = Rw.T
            t_w2c = -R_w2c @ p_shift
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = R_w2c
            viewmat[:3, 3] = t_w2c

            img, _, _ = rasterization(
                means=params["means"], quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors,
                viewmats=torch.tensor(viewmat, device=device)[None],
                Ks=K, width=W, height=H, sh_degree=sh_degree,
                camera_model="pinhole", with_ut=True, with_eval3d=True,
                packed=False, radial_coeffs=radial,
                near_plane=0.01, far_plane=100.0)
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(frame, f"t={r['t']:5.1f}s  goal {r['goal_dist_m']:5.1f}m  "
                               f"crowd {r['n_obstacles']:2d}"
                               + ("  STOP" if r["stopped"] else ""),
                        (14, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(frame)
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames", flush=True)
    vw.release()
    print(f"wrote {args.out} ({n} frames at {W}x{H})")


main()
