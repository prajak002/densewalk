"""Place GEM-X SOMA bodies in the metric Varanasi world and colour them from the video.

Per person (gemx_people.py output, camera-frame meshes with the calibrated K):
  rotation  R_world<-cam = canon.R @ R_c2w(COLMAP, that frame) -- the validated camera pose
  position  pelvis xy = the person's metric track (crowd_world_v2) at that frame; feet on z=0
            (people without a metric track use GEM-X's own depth, flagged in meta)
  colour    each vertex takes the video colour from the sampled frame where it faces the
            camera most directly (x2 frames from the x4 super-resolved set)
Cross-check reported: horizontal distance between GEM-X's own depth placement and the track.

Web format: faces.bin (uint32), per person <tid>.bin = per stored frame [pelvis xyz f32] +
int16 vertex offsets from pelvis in mm, colors (uint8 RGB), meta.json.
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

FPS = 29.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemx", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--crowd", type=Path, required=True)
    ap.add_argument("--sr-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th video frame (viewer interpolates)")
    ap.add_argument("--people", type=Path, default=None,
                    help="sprite dir (sprites.json, match.json, sprites/): colour only inside the person's own instance mask")
    args = ap.parse_args()
    w = json.loads(args.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))
    _, imgs, _, _ = read_colmap(args.model_dir)
    c2w = {int(im["name"].split(".")[0]): im["w2c"].inverse() for im in imgs}
    crowd = json.loads(args.crowd.read_text())["tracks"]
    faces = np.load(args.gemx / "faces.npy").astype(np.uint32)
    args.out.mkdir(parents=True, exist_ok=True)
    faces.tofile(args.out / "faces.bin")
    masks = {}
    if args.people is not None:
        spr = json.loads((args.people / "sprites.json").read_text())["tracks"]
        for mtid, mm in json.loads((args.people / "match.json").read_text()).items():
            masks[mtid] = {d["frame"]: d for d in spr[mm["sprite_id"]]}

    meta, xcheck = {"n_faces": int(len(faces)), "fps": FPS, "people": {}}, []
    for f in sorted(args.gemx.glob("*.npz"), key=lambda p: int(p.stem)):
        tid = f.stem
        z = np.load(f)
        fr, V, J, K = z["frames"], z["verts"].astype(np.float32), z["joints"], z["K"]
        cs = float(z["clip_scale"])
        tr = crowd.get(tid)
        keep, P, Vrel = [], [], []
        for k, fi in enumerate(fr):
            if fi not in c2w:
                continue
            Rwc = canon.R @ c2w[int(fi)].R
            pel = J[k, 0]
            vr = (V[k] - pel) @ Rwc.T
            gem_xy = (canon.apply(c2w[int(fi)].apply((pel / scale)[None]))[0] * scale)[:2]
            if tr is not None:
                ts = np.array(tr["t_s"]); xy = np.array(tr["xy_m"])
                tt = fi / FPS
                if tt < ts[0] - 0.05 or tt > ts[-1] + 0.05:
                    continue
                pxy = np.array([np.interp(tt, ts, xy[:, 0]), np.interp(tt, ts, xy[:, 1])])
                xcheck.append(float(np.linalg.norm(gem_xy - pxy)))
            else:
                pxy = gem_xy
            keep.append(k); P.append(pxy); Vrel.append(vr)
        if len(keep) < 6:
            continue
        keep = np.array(keep); P = np.array(P); Vrel = np.array(Vrel)
        # feet on the floor: pelvis height = -(lowest vertex), median-smoothed over ~0.3 s
        low = Vrel[:, :, 2].min(1)
        pad = np.pad(low, 4, mode="edge")
        low_s = np.array([np.median(pad[i:i + 9]) for i in range(len(low))])
        pel_z = -low_s

        # colour from the video: per vertex, the sampled frame where it faces the camera most
        n = len(keep)
        samp = np.unique(np.linspace(0, n - 1, min(16, n)).astype(int))
        best = np.full(V.shape[1], -1.0); col = np.zeros((V.shape[1], 3), np.float32)
        for s in samp:
            k = keep[s]; fi = int(fr[k])
            img = cv2.imread(str(args.sr_frames / f"{fi:05d}.jpg"))
            if img is None:
                continue
            img = cv2.resize(img, (int(640 * cs), int(360 * cs)), interpolation=cv2.INTER_AREA)
            v = V[k]
            nrm = np.zeros_like(v)
            tri = v[faces]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for c in range(3):
                np.add.at(nrm, faces[:, c], fn)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            view = -v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            facing = (nrm * view).sum(1)
            uv = (K @ v.T).T; uv = uv[:, :2] / uv[:, 2:3]
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
            md = masks.get(tid, {}).get(fi)
            if masks and md is None:
                continue                       # no instance mask this frame: skip rather than risk bleed
            if md is not None:
                a = cv2.imread(str(args.people / "sprites" / md["file"]), cv2.IMREAD_UNCHANGED)[:, :, 3]
                bx0, by0 = int(round(max(0, md["xyxy"][0]))), int(round(max(0, md["xyxy"][1])))
                ox, oy = (uv[:, 0] / cs).astype(int) - bx0, (uv[:, 1] / cs).astype(int) - by0
                inm = (ox >= 0) & (oy >= 0) & (ox < a.shape[1]) & (oy < a.shape[0])
                inm[inm] = a[oy[inm], ox[inm]] > 127
                ok &= inm
            upd = ok & (facing > best)
            px = img[uv[upd, 1].astype(int), uv[upd, 0].astype(int)][:, ::-1].astype(np.float32)
            col[upd] = px; best[upd] = facing[upd]
        # vertices never seen facing the camera: nearest seen vertex's colour
        seen = best > 0.05
        if seen.sum() > 20 and (~seen).any():
            ref = Vrel[0][seen]; q = Vrel[0][~seen]
            idx = np.array([np.argmin(((ref - p) ** 2).sum(1)) for p in q])
            col[~seen] = col[seen][idx]

        sel = np.arange(0, n, args.stride)
        off = np.clip(np.round(Vrel[sel] * 1000), -32767, 32767).astype(np.int16)
        pel = np.concatenate([P[sel], pel_z[sel, None]], 1).astype(np.float32)
        with open(args.out / f"{tid}.bin", "wb") as fh:
            fh.write(pel.tobytes()); fh.write(off.tobytes()); fh.write(np.clip(col, 0, 255).astype(np.uint8).tobytes())
        meta["people"][tid] = {"frames": fr[keep][sel].astype(int).tolist(), "n_verts": int(V.shape[1]),
                               "anchor": "track" if tr is not None else "gem_depth",
                               "height_m": round(float(np.median(Vrel[:, :, 2].max(1) - Vrel[:, :, 2].min(1))), 2)}
    meta["xcheck_gem_depth_vs_track_m"] = np.percentile(xcheck, [25, 50, 75]).round(2).tolist() if xcheck else None
    hs = [p["height_m"] for p in meta["people"].values()]
    meta["height_m_p10_50_90"] = np.percentile(hs, [10, 50, 90]).round(2).tolist() if hs else None
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k != "people"} | {"n_people": len(meta["people"]),
          "anchors": {a: sum(p["anchor"] == a for p in meta["people"].values()) for a in ("track", "gem_depth")}}, indent=2))


if __name__ == "__main__":
    main()
