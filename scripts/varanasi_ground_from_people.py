"""M1-Varanasi step 7b: recover the ground plane from the crowd, not the road.

Fitting the plane to SfM points fails on this clip: the road is almost never
unoccluded, so only 777 of 6198 candidate points supported it, and the result
was tilted enough that reconstructed pedestrian height correlated 0.871 with
distance -- far people came out 4x taller than near ones.

So invert the instrument. People are vertical and roughly one height, and this
clip has plenty of them. Solve for the plane (normal + offset) that makes
reconstructed stature *independent of depth*, then set the scale so the median
is ADULT_STATURE_M. The residual depth-correlation after fitting is reported as
the honest quality measure: near zero means the plane is right, and anything
else means it is not, stated rather than hidden.

The road we cannot see is never used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames
from varanasi_train_splat import read_colmap

ADULT_STATURE_M = 1.65
CAMERA_HEIGHT_M = 1.55   # handheld phone, chest-to-eye level


def gather(model_dir: Path, tracks_path: Path):
    cam, imgs, xyz, _ = read_colmap(model_dir)
    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    poses = {im["name"]: im["w2c"] for im in imgs}
    tracks = json.loads(tracks_path.read_text())["tracks"]

    origins, rays_f, rays_h, tids = [], [], [], []
    for tid, dets in tracks.items():
        if dets[0]["cls"] != "person":
            continue
        for det in dets:
            w2c = poses.get(det["image"])
            if w2c is None:
                continue
            c2w = w2c.inverse()
            o = c2w.apply(np.zeros((1, 3)))[0]
            x0, y0, x1, y1 = det["xyxy"]
            xm = (x0 + x1) / 2
            rf = c2w.R @ (Kinv @ np.array([xm, y1, 1.0]))
            rh = c2w.R @ (Kinv @ np.array([xm, y0, 1.0]))
            origins.append(o)
            rays_f.append(rf / np.linalg.norm(rf))
            rays_h.append(rh / np.linalg.norm(rh))
            tids.append(tid)
    return (np.array(origins), np.array(rays_f), np.array(rays_h),
            np.array(tids), imgs, xyz)


def heights_for_plane(n, d, O, RF, RH):
    """Stature of each observation under a candidate plane; NaN where the
    geometry is degenerate (ray parallel to the plane, or behind the camera)."""
    den = RF @ n
    t = -(O @ n + d) / np.where(np.abs(den) < 1e-9, np.nan, den)
    t = np.where(t > 0, t, np.nan)
    G = O + t[:, None] * RF
    cn = np.cross(np.broadcast_to(n, RH.shape), RH)
    dd = (cn * cn).sum(axis=1)
    h = -((np.cross(G - O, RH) * cn).sum(axis=1)) / np.where(dd < 1e-12, np.nan, dd)
    return h, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    O, RF, RH, tids, imgs, xyz = gather(args.model_dir, args.tracks)
    print(f"{len(O)} person observations from {len(set(tids))} tracks")
    centres = np.stack([im["w2c"].inverse().apply(np.zeros((1, 3)))[0] for im in imgs])
    up0 = -np.median(np.stack([im["w2c"].R[1] for im in imgs]), axis=0)
    up0 /= np.linalg.norm(up0)
    d0 = -float(np.median(centres @ up0)) + 1.9

    def pack(n, d):
        return np.array([n[0], n[1], n[2], d])

    def unpack(p):
        n = p[:3] / np.linalg.norm(p[:3])
        return n, p[3]

    def resid(p):
        n, d = unpack(p)
        h, t = heights_for_plane(n, d, O, RF, RH)
        ok = np.isfinite(h) & np.isfinite(t) & (h > 0.05) & (h < 10)
        # the residual vector must keep a fixed length across calls, including
        # on degenerate candidates, or least_squares refuses to run
        if ok.sum() < 50:
            return np.full(len(O) + 400, 10.0)
        hh, tt = h[ok], t[ok]
        med = np.median(hh)
        # residual 1: every stature should equal the median (depth-independent)
        r = np.zeros(len(O))
        r[ok] = (hh - med) / max(med, 1e-6)
        # residual 2: explicitly punish any linear trend with depth
        tt_c = (tt - tt.mean()) / max(tt.std(), 1e-6)
        slope = float((tt_c * ((hh - med) / max(med, 1e-6))).mean())
        # residual 3: break the offset degeneracy INSIDE the solve. Stature
        # consistency alone fixes only the plane's tilt -- slide it down and
        # shrink the scale and nothing changes -- so require the camera-height
        # scale and the stature scale to agree. Solving this after the tilt
        # instead of with it lets the two constraints fight and lands in a
        # degenerate root (it gave a 9 m/s walk).
        cam_u = float(np.median(centres @ n + d))
        if cam_u <= 1e-6:
            return np.full(len(O) + 400, 10.0)
        gap = (CAMERA_HEIGHT_M / cam_u - ADULT_STATURE_M / max(med, 1e-6)) / \
              max(ADULT_STATURE_M / max(med, 1e-6), 1e-6)
        return np.concatenate([r, np.full(200, slope * 5.0), np.full(200, gap * 3.0)])

    sol = least_squares(resid, pack(up0, d0), method="lm", max_nfev=4000)
    n, d = unpack(sol.x)
    if n @ up0 < 0:
        n, d = -n, -d

    h, t = heights_for_plane(n, d, O, RF, RH)
    ok = np.isfinite(h) & np.isfinite(t) & (h > 0.05) & (h < 10)
    med_h = float(np.median(h[ok]))
    corr = float(np.corrcoef(t[ok], h[ok])[0, 1])
    scale = ADULT_STATURE_M / med_h
    cam_h = float(np.median(centres @ n + d)) * scale

    print(f"plane n={n.round(4).tolist()} d={d:.4f}")
    print(f"median stature {med_h:.4f} units -> scale {scale:.4f} m/unit")
    print(f"residual corr(depth, height) = {corr:+.3f}   (was +0.871)")
    print(f"implied camera height = {cam_h:.2f} m")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 10), (10, 40)]:
        m = ok & (t >= lo) & (t < hi)
        if m.sum() > 20:
            print(f"  depth {lo:>2}-{hi:<2}: n={int(m.sum()):5d} "
                  f"median height = {np.median(h[m]) * scale:.2f} m")

    z = n
    x = centres[-1] - centres[0]
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    canon = frames.Pose(np.stack([x, y, z]), np.array([0.0, 0.0, d]))
    C = canon.apply(centres) * scale

    out = {
        "method": "ground plane solved from pedestrian verticality, not from SfM road points",
        "scale_m_per_unit": scale,
        "plane_normal": n.tolist(), "plane_d": float(d),
        "canonical_R": canon.R.tolist(), "canonical_t": canon.t.tolist(),
        "median_stature_units": med_h,
        "residual_depth_height_corr": corr,
        "camera_height_m": cam_h,
        "path_length_m": float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()),
        "walking_speed_mps": float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum() / 12.045),
        "n_observations": int(ok.sum()),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("canonical_R", "canonical_t", "plane_normal")}, indent=2))


if __name__ == "__main__":
    main()
