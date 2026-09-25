"""Export the trained Varanasi splat as a cleaned 3DGS .ply in the canonical metric frame.

COLMAP-shifted splat -> metric (metres, Z-up, floor z=0) through world.json and frames.Pose:
    p_m = scale * canon(p_colmap_shifted + origin),  scales * scale,  q = q_canon * q
Only the SH DC term is written: rotating higher SH bands into the new frame is not
done here, and DC keeps the colour without view-dependent effects.

Cleaning, each count reported:
  low opacity     sigmoid(opacity) < --min-opacity (near-invisible floaters)
  oversized       largest axis > --max-scale-m (sky/background blobs)
  below floor     z < --floor-tol (ghost geometry under the street, see varanasi memory)
  far             > --max-dist-m from the camera path (triangulation noise)
  crowd ghosts    inside the walkable corridor and --ghost-zmin < z < --ghost-zmax: the
                  masked crowd leaves unsupervised gaussians exactly there; people are
                  re-inserted as SOMA bodies instead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--crowd", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--max-scale-m", type=float, default=1.5)
    ap.add_argument("--floor-tol", type=float, default=-0.25)
    ap.add_argument("--max-dist-m", type=float, default=30.0)
    ap.add_argument("--ghost-zmin", type=float, default=0.12)
    ap.add_argument("--ghost-zmax", type=float, default=2.3)
    ap.add_argument("--ghost-margin-m", type=float, default=0.3, help="shrink the corridor so shopfront edges survive")
    ap.add_argument("--no-ghost-cut", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    origin = torch.load(str(args.ckpt) + ".origin.pt", map_location="cpu")["origin"].numpy().astype(np.float64)
    w = json.loads(args.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    means = ck["means"].numpy().astype(np.float64) + origin
    xyz = canon.apply(means) * scale
    log_s = ck["scales"].numpy().astype(np.float64) + np.log(scale)
    q_wxyz = ck["quats"].numpy().astype(np.float64)
    q_wxyz /= np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    rot = Rotation.from_matrix(canon.R) * Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    q_new = rot.as_quat()[:, [3, 0, 1, 2]]                  # back to wxyz
    opac_logit = ck["opacities"].numpy().astype(np.float64)
    dc = ck["sh0"].numpy().reshape(-1, 3).astype(np.float64)

    n0 = len(xyz)
    keep = np.ones(n0, bool)
    report = {}

    def cut(name, m):
        nonlocal keep
        report[name] = int((keep & m).sum())
        keep &= ~m

    cut("low_opacity", 1 / (1 + np.exp(-opac_logit)) < args.min_opacity)
    cut("oversized", np.exp(log_s).max(1) > args.max_scale_m)
    cut("below_floor", xyz[:, 2] < args.floor_tol)
    cam = np.array(json.loads(args.crowd.read_text())["summary"]["camera_path_m"])
    d = np.min(np.linalg.norm(xyz[:, None, :2] - cam[None, ::5], axis=2), axis=1)
    cut("far", d > args.max_dist_m)
    if not args.no_ghost_cut:
        run = json.loads(args.run.read_text())["summary"]
        yaw0 = run["spawn_yaw"]
        S2W = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]]).T
        ys = sorted((S2W @ np.array([0.0, yc]) + np.array(run["start_xy"]))[1] for yc in run["corridor_y"])
        lo, hi = ys[0] + args.ghost_margin_m, ys[1] - args.ghost_margin_m
        xs = cam[:, 0]
        ghost = ((xyz[:, 1] > lo) & (xyz[:, 1] < hi) & (xyz[:, 2] > args.ghost_zmin) & (xyz[:, 2] < args.ghost_zmax)
                 & (xyz[:, 0] > xs.min() - 5) & (xyz[:, 0] < xs.max() + 15))
        cut("crowd_ghost_corridor", ghost)
        report["corridor_world_y"] = [lo, hi]
    report["kept"] = int(keep.sum()); report["input"] = n0

    P = xyz[keep]; S = log_s[keep]; Q = q_new[keep]; O = opac_logit[keep]; D = dc[keep]
    n = len(P)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([P, np.zeros((n, 3)), D, O[:, None], S, Q], 1).astype(np.float32)
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
             "".join(f"property float {p}\n" for p in props) + "end_header\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header.encode()); f.write(data.tobytes())
    report["floor_z_percentiles_kept"] = np.percentile(P[:, 2], [1, 5, 50]).round(3).tolist()
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
