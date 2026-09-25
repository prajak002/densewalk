"""M4 composite: put the G1 into the reconstructed Varanasi street.

The robot layer comes from Isaac (RGBA, alpha from depth, ground hidden); the
street comes from the splat. Both are rendered from the SAME camera pose, so
compositing them is a straight alpha blend rather than a visual trick -- the
robot is where it actually walked, seen from where the chase camera actually
was, inside the street that was actually reconstructed.

The splat must be rendered with ISAAC's intrinsics, not COLMAP's, or the two
layers disagree about field of view and the robot floats at the wrong scale:
    fx = width * focal_length_mm / horizontal_aperture_mm
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

CAM_BACK_M = 2.6        # must match the mount offset in varanasi_m4_capture.py
CAM_UP_M = 1.25
CAM_PITCH_DEG = 20.0
FOCAL_MM = 18.0
H_APERTURE_MM = 20.955  # Isaac PinholeCameraCfg default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cap-dir", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda"
    w = json.loads(args.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    meta = json.loads((args.cap_dir / "meta.json").read_text())
    start_xy = np.array(meta["start_xy"])
    yaw0 = float(meta["spawn_yaw"])
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    S2W = np.array([[c0, s0], [-s0, c0]])      # sim -> canonical
    W, H = meta["width"], meta["height"]

    ck = torch.load(args.ckpt, map_location=device)
    origin = torch.load(str(args.ckpt) + ".origin.pt")["origin"].numpy().astype(np.float64)
    sh_degree = int(ck["sh_degree"])
    params = {k: ck[k].to(device) for k in
              ("means", "scales", "quats", "opacities", "sh0", "shN")}
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)

    fx = W * FOCAL_MM / H_APERTURE_MM
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]
    print(f"{params['means'].shape[0]} gaussians; splat rendered at "
          f"Isaac intrinsics fx={fx:.1f} ({W}x{H})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("could not open VideoWriter")

    pitch = np.deg2rad(CAM_PITCH_DEG)
    n = 0
    with torch.no_grad():
        for fr in meta["frames"]:
            robot_png = args.cap_dir / "robot" / f"{fr['i']:05d}.png"
            layer = cv2.imread(str(robot_png), cv2.IMREAD_UNCHANGED)
            if layer is None:
                continue

            # robot pose: sim -> canonical metric
            xy = S2W @ np.array(fr["robot_xy_sim"]) + start_xy
            yaw = fr["yaw_sim"] + yaw0
            fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            cam_pos = np.array([xy[0], xy[1], fr["robot_z"]]) \
                - CAM_BACK_M * fwd + np.array([0, 0, CAM_UP_M])

            # look direction: forward, pitched down
            look = np.array([np.cos(yaw) * np.cos(pitch),
                             np.sin(yaw) * np.cos(pitch),
                             -np.sin(pitch)])
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(look, up); right /= np.linalg.norm(right)
            down = np.cross(look, right)
            R_c2w_canon = np.stack([right, down, look], axis=1)  # OpenCV axes

            p_colmap = canon.inverse().apply((cam_pos / scale)[None])[0] - origin
            Rw = canon.R.T @ R_c2w_canon
            R_w2c = Rw.T
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = R_w2c
            viewmat[:3, 3] = -R_w2c @ p_colmap

            img, _, _ = rasterization(
                means=params["means"], quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors,
                viewmats=torch.tensor(viewmat, device=device)[None],
                Ks=K, width=W, height=H, sh_degree=sh_degree,
                camera_model="pinhole", with_ut=True, with_eval3d=True,
                packed=False, near_plane=0.01, far_plane=100.0)
            street = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            street = cv2.cvtColor(street, cv2.COLOR_RGB2BGR)

            a = (layer[..., 3:4].astype(np.float32) / 255.0)
            rob = layer[..., :3].astype(np.float32)          # BGR
            # Kill green spill. Anti-aliased silhouette pixels are a blend of
            # robot and the green background, so the cutout reads with a lurid
            # green rim. The G1 is grey/white, so clamping green to the other
            # channels removes the rim without touching the robot itself.
            rob[..., 1] = np.minimum(rob[..., 1],
                                     np.maximum(rob[..., 0], rob[..., 2]))
            # erode by a pixel before feathering so no pure-background pixel
            # survives into the blend
            am = cv2.erode((a[..., 0] * 255).astype(np.uint8), np.ones((3, 3), np.uint8))
            a = cv2.GaussianBlur(am.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
            comp = (rob * a + street.astype(np.float32) * (1 - a)).astype(np.uint8)

            cv2.rectangle(comp, (0, H - 46), (W, H), (0, 0, 0), -1)
            cv2.putText(comp, f"t={fr['t']:5.1f}s   goal {fr['goal_dist_m']:5.1f} m   "
                              f"crowd {fr['n_obstacles']:2d}"
                              + ("   HOLDING" if fr["stopped"] else ""),
                        (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(comp)
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames", flush=True)
    vw.release()
    print(f"wrote {args.out} ({n} frames)")


main()
