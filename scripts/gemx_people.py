"""Full 3D SOMA body per real person in the Varanasi clip, with GEM-X (NVlabs, video -> SOMA 77-joint).

GEM-X handles one person per video, driven by that person's per-frame boxes
(preprocess/bbx.pt; when present, its own detector is skipped). So, per tracked
person (tracks.json, the same tracks crowd_world.json was built from):

  1. sub-clip of the frames they are visible in, from the x4 super-resolved frames
     downscaled to 1280x720 (x2 of the original, sharper keypoints than the 640x360 source)
  2. bbx.pt from their own boxes (gaps interpolated, smoothed as GEM-X does)
  3. GEM-X's own run_preprocess + load_data_dict + model.predict, static camera
     (camera motion is taken from our COLMAP poses later, not GEM-X's)
     with the calibrated COLMAP intrinsics in place of GEM-X's size-based K guess
  4. SomaLayer on body_params_incam -> per-frame mesh vertices in the camera frame

Uses GEM-X functions directly (scripts/demo/demo_soma.py) rather than re-implementing
them; only the preview video rendering is skipped.
Output per person: <out>/<tid>.npz with frames, verts (float16, camera frame, metres), joints.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

GEMX = Path("/workspace/GEM-X")
sys.path.insert(0, str(GEMX))
spec = importlib.util.spec_from_file_location("demo_soma", GEMX / "scripts/demo/demo_soma.py")
demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
import hydra  # noqa: E402
from gem.utils.geo_transform import get_bbx_xys_from_xyxy  # noqa: E402
from gem.utils.kp2d_utils import smooth_bbx_xyxy  # noqa: E402
from gem.utils.net_utils import detach_to_cpu, to_cuda  # noqa: E402
from gem.utils.soma_utils.soma_layer import SomaLayer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--sr-frames", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated track ids")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--scale", type=float, default=2.0, help="clip pixels per original pixel")
    ap.add_argument("--focal-px", type=float, default=445.1, help="COLMAP focal length at 640x360 (scaled with the clip)")
    args = ap.parse_args()
    tracks = json.loads(args.tracks.read_text())["tracks"]
    ids = [t for t in tracks if tracks[t][0]["cls"] == "person"]
    if args.only:
        ids = [t for t in ids if t in args.only.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    W, H = int(640 * args.scale), int(360 * args.scale)

    model, soma, faces_saved = None, None, False
    for tid in ids:
        if (args.out / f"{tid}.npz").exists():
            continue
        dets = sorted(tracks[tid], key=lambda d: d["frame"])
        f0, f1 = dets[0]["frame"], dets[-1]["frame"]
        if f1 - f0 + 1 < args.min_frames:
            continue
        t0 = time.time()
        fr = np.arange(f0, f1 + 1)
        known = np.array([d["frame"] for d in dets])
        box = np.array([d["xyxy"] for d in dets], dtype=np.float64)
        xyxy = np.stack([np.interp(fr, known, box[:, k]) for k in range(4)], 1) * args.scale

        name = f"p{tid}"
        vdir = args.work / name; (vdir / name / "preprocess").mkdir(parents=True, exist_ok=True)
        vpath = vdir / f"{name}_src.mp4"
        vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for f in fr:
            im = cv2.imread(str(args.sr_frames / f"{f:05d}.jpg"))
            vw.write(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
        vw.release()

        cargs = Namespace(video=str(vpath), output_root=str(vdir), static_cam=True, verbose=False,
                          render_mhr=False, sam3d_ckpt_path=None, sam3d_mhr_path=None, ckpt=None,
                          exp="gem_soma_regression", retarget=False)
        cfg = demo._build_cfg(cargs)
        demo._copy_video_if_needed(cfg)
        bb = torch.from_numpy(xyxy).float()
        bb = smooth_bbx_xyxy(bb, window=5)
        bb[:, [0, 2]] = bb[:, [0, 2]].clamp(0, W - 1); bb[:, [1, 3]] = bb[:, [1, 3]].clamp(0, H - 1)
        torch.save({"bbx_xyxy": bb, "bbx_xys": get_bbx_xys_from_xyxy(bb, base_enlarge=1.2).float()}, cfg.paths.bbx)
        demo.run_preprocess(cfg)
        data = demo.load_data_dict(cfg)
        # GEM-X's demo guesses K from the image size (f = max(W,H)); use the calibrated phone intrinsics instead
        Kc = torch.tensor([[args.focal_px * args.scale, 0, W / 2], [0, args.focal_px * args.scale, H / 2], [0, 0, 1]], dtype=torch.float32)
        data["K_fullimg"] = Kc[None].repeat(len(fr), 1, 1)
        if model is None:
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(demo.resolve_ckpt_path(cfg))
            model = model.eval().cuda()
        with torch.no_grad():
            pred = detach_to_cpu(model.predict(data, static_cam=True, postproc=True))
        torch.save(pred, cfg.paths.hpe_results)
        bp = demo._get_body_params(pred, "body_params_incam")
        if soma is None:
            soma = SomaLayer(data_root=str(GEMX / "inputs/soma_assets"), low_lod=True, device="cuda:0",
                             identity_model_type="mhr", mode="warp")
        with torch.no_grad():
            o = soma(**to_cuda(bp))
        verts = o["vertices"].float().cpu().numpy(); joints = o["joints"].float().cpu().numpy()
        if not faces_saved:
            np.save(args.out / "faces.npy", soma.faces.cpu().numpy().astype(np.int32)); faces_saved = True
        np.savez_compressed(args.out / f"{tid}.npz", frames=fr, verts=verts.astype(np.float16), joints=joints.astype(np.float32),
                            K=np.asarray(pred["K_fullimg"][0]), clip_scale=args.scale)
        print(f"[{tid}] frames {f0}-{f1} ({len(fr)})  verts {verts.shape}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
