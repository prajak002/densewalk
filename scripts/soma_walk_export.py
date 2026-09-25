"""Export a SOMA body + Kimodo walk cycle for the three.js viewer (outputs/scene3d/soma/).

Body   Kimodo's somaskel77 skin (kimodo/assets/skeletons/somaskel77/skin_standard.npz):
       the SOMA mid-LOD mesh, 18,056 vertices, bind transforms for the 77-joint rig.
Motion Kimodo's shipped SOMA example "A person is casually walking forward slowly"
       (kimodo-soma-rp/05_root_path/motion.npz, 30 fps, somaskel30). Expanded to 77
       joints exactly as kimodo.viz.soma_skin.SOMASkin.skin does it.

The viewer re-implements skinning with three.js SkinnedMesh, which is the same
linear-blend skinning SOMASkin.lbs computes (world joint transform @ inverse bind).
Checks, printed and written to validation.json:
  1. our numpy LBS (all weights) vs SOMASkin.skin itself, max vertex error
  2. three.js uses 4 influences: error of top-4 renormalised weights vs all
  3. loop seam: pose distance between loop start and end
  4. foot contact: lowest foot joint height over the loop (floor at 0)

Axes: Kimodo is Y-up, +Z forward, metres. Converted once, through frames.Pose,
to the project's Z-up with the body facing +X (x_zup = z_k, y_zup = x_k, z_zup = y_k).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from densewalk import frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimodo", type=Path, default=Path("/workspace/kimodo"))
    ap.add_argument("--motion", default="kimodo/assets/demo/examples/kimodo-soma-rp/05_root_path/motion.npz")
    ap.add_argument("--out", type=Path, default=Path("soma_out"))
    ap.add_argument("--skip-s", type=float, default=1.5, help="ignore the start-up from standing")
    args = ap.parse_args()
    sys.path.insert(0, str(args.kimodo))
    from kimodo.assets import SKELETONS_ROOT
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77, batch_rigid_transform, global_rots_to_local_rots
    # import the skin module by file so kimodo.viz's GUI deps are not needed
    spec = importlib.util.spec_from_file_location("soma_skin", args.kimodo / "kimodo/viz/soma_skin.py")
    soma_skin = importlib.util.module_from_spec(spec); spec.loader.exec_module(soma_skin)

    root = Path(SKELETONS_ROOT)
    sk30 = SOMASkeleton30(str(root / "somaskel30"))
    sk77 = SOMASkeleton77(str(root / "somaskel77"))
    skin = soma_skin.SOMASkin(sk30)
    z = np.load(args.kimodo / args.motion)
    g30 = torch.tensor(z["global_rot_mats"], dtype=torch.float32)
    p30 = torch.tensor(z["posed_joints"], dtype=torch.float32)
    fps = 30.0
    nF = len(g30)

    # --- 30 -> 77, exactly as SOMASkin.skin ---
    local30 = global_rots_to_local_rots(g30, sk30)
    local77 = sk30.to_SOMASkeleton77(local30)
    neutral = sk77.neutral_joints[None].repeat(nF, 1, 1)
    jpos, grot = batch_rigid_transform(local77, neutral, sk77.joint_parents, sk77.root_idx)
    jpos = jpos + p30[:, sk30.root_idx:sk30.root_idx + 1]
    T = np.tile(np.eye(4, dtype=np.float64), (nF, 77, 1, 1))
    T[..., :3, :3] = grot.numpy(); T[..., :3, 3] = jpos.numpy()

    S = np.load(root / "somaskel77" / "skin_standard.npz")
    Vb = S["bind_vertices"].astype(np.float64); F = S["faces"]
    Bi = np.linalg.inv(S["bind_rig_transform"].astype(np.float64))
    li, lw = S["lbs_indices"], S["lbs_weights"].astype(np.float64)

    def lbs(Tf, idx, w):
        A = Tf @ Bi                                           # [77,4,4]
        vh = np.concatenate([Vb, np.ones((len(Vb), 1))], 1)   # [V,4]
        out = np.zeros((len(Vb), 3))
        for k in range(idx.shape[1]):
            out += w[:, k:k + 1] * np.einsum("vij,vj->vi", A[idx[:, k], :3, :], vh)
        return out

    # check 1: our LBS == SOMASkin.skin
    fchk = [0, nF // 2, nF - 1]
    ref = skin.skin(g30[fchk], p30[fchk], rot_is_global=True).numpy()
    e1 = max(np.abs(lbs(T[f], li, lw) - ref[i]).max() for i, f in enumerate(fchk))
    # check 2: top-4 renormalised (three.js limit)
    order = np.argsort(-lw, axis=1)[:, :4]
    i4 = np.take_along_axis(li, order, 1); w4 = np.take_along_axis(lw, order, 1)
    w4 = w4 / w4.sum(1, keepdims=True)
    e2 = [np.linalg.norm(lbs(T[f], i4, w4) - lbs(T[f], li, lw), axis=1) for f in fchk]
    e2_max, e2_mean = float(max(e.max() for e in e2)), float(np.mean([e.mean() for e in e2]))

    # --- loop: most similar pose pair a<b, 1-5 s apart, after start-up ---
    a0 = int(args.skip_s * fps)
    L = local77.numpy().reshape(nF, -1)
    best = (1e9, 0, 0)
    for a in range(a0, nF - 30):
        for b in range(a + 30, min(nF, a + 150)):
            d = np.linalg.norm(L[a] - L[b])
            if d < best[0]:
                best = (d, a, b)
    seam, fa, fb = best
    root_xz = jpos[:, sk77.root_idx, [0, 2]].numpy()
    disp = root_xz[fb] - root_xz[fa]
    clip_speed = float(np.linalg.norm(disp) / ((fb - fa) / fps))
    heading = float(np.arctan2(disp[0], disp[1]))          # angle of travel from +Z about +Y

    # in-place: remove horizontal root progression, rotate travel direction onto +Z
    c, s = np.cos(-heading), np.sin(-heading)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # rotation about +Y by -heading
    Tl = T[fa:fb].copy()
    Tl[..., :3, :3] = Ry @ Tl[..., :3, :3]
    Tl[..., :3, 3] = Tl[..., :3, 3] @ Ry.T
    rz = Tl[:, sk77.root_idx, :3, 3].copy()
    Tl[..., :3, 3] -= np.stack([rz[:, 0], np.zeros(len(rz)), rz[:, 2]], 1)[:, None, :]

    # check 4: feet on the floor
    feet = [sk77.bone_index[n] for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")]
    foot_min = float(Tl[:, feet, 1, 3].min())

    # --- Y-up -> Z-up (facing +X), once, through frames.Pose ---
    C = frames.Pose(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64), np.zeros(3))
    Cm = C.as_matrix()
    Tz = Cm @ Tl                                            # posed transforms
    Bz = Cm @ S["bind_rig_transform"].astype(np.float64)    # bind transforms
    Vz = C.apply(Vb)                                        # bind vertices

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "soma_body.bin", "wb") as f:
        f.write(Vz.astype(np.float32).tobytes())
        f.write(F.astype(np.uint32).tobytes())
        f.write(i4.astype(np.uint16).tobytes())
        f.write(w4.astype(np.float32).tobytes())
        f.write(np.linalg.inv(Bz).astype(np.float32).transpose(0, 2, 1).tobytes())  # column-major for three.js
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Tz[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(Tz), 77, 4)  # xyzw
    anim = np.concatenate([Tz[..., :3, 3], q], -1).astype(np.float32)  # [frames, 77, 7]
    anim.tofile(args.out / "soma_walk.bin")
    meta = {
        "source_body": "kimodo somaskel77 skin_standard.npz (SOMA mid LOD)",
        "source_motion": args.motion + " -- 'A person is casually walking forward slowly'",
        "fps": fps, "n_frames": int(len(Tz)), "n_joints": 77,
        "n_vertices": int(len(Vz)), "n_faces": int(len(F)),
        "clip_speed_mps": clip_speed, "loop_frames": [fa, fb],
        "frame": "Z-up, body faces +X, metres; hips horizontally fixed (in-place loop)",
        "validation": {
            "lbs_vs_kimodo_SOMASkin_max_m": float(e1),
            "top4_weights_vs_full_max_m": e2_max, "top4_weights_vs_full_mean_m": e2_mean,
            "loop_seam_pose_dist": float(seam),
            "lowest_foot_joint_z_m": foot_min,
        },
    }
    (args.out / "soma_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
