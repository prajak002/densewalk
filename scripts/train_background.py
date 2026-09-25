"""Step 5: train a static-background 3DGUT model on an AV2 sequence.

Uses the calibrated AV2 poses directly (no SfM). Dynamic objects are
excluded from the loss via the SAM2 masks from Step 4, so the model fits
background only. Gaussians are initialised from the sequence's own lidar
sweeps rather than randomly.

3DGUT is enabled via gsplat's with_ut=True / with_eval3d=True, with AV2's
radial distortion passed through so the real (distorted) pixels are fit
without undistorting the images first.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

from densewalk import frames
from densewalk.av2 import Av2Sequence, RING_CAMERAS, assert_av2_frame_sane


# Cameras that see the ego vehicle's own bodywork. It sits at a fixed
# image position but a different world position every frame, so fitting it
# produces floaters -- exclude those pixels from the loss.
EGO_BODY_MASK_FRAC = {
    "ring_front_center": 0.16,
    "ring_rear_left": 0.13,
    "ring_rear_right": 0.13,
}


def load_views(seq, cam_names, mask_dir: Path, downscale: int, stride: int):
    """Build the per-image view list: image, mask, K, world->cam viewmat."""
    views = []
    for cam_name in cam_names:
        cam = seq.camera(cam_name)
        timestamps = seq.image_timestamps(cam_name)[::stride]
        for ts in timestamps:
            img = cv2.imread(str(seq.image_path(cam_name, ts)))
            if img is None:
                continue
            m_path = mask_dir / cam_name / f"{ts}.png"
            mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], np.uint8)
            if downscale > 1:
                img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale),
                                 interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            ego_frac = EGO_BODY_MASK_FRAC.get(cam_name, 0.0)
            if ego_frac > 0:
                cut = int(mask.shape[0] * (1.0 - ego_frac))
                mask[cut:, :] = 255
            s = 1.0 / downscale
            K = np.array([[cam.intr.fx * s, 0, cam.intr.cx * s],
                          [0, cam.intr.fy * s, cam.intr.cy * s],
                          [0, 0, 1]], dtype=np.float32)
            w2c = seq.world_to_cam(cam, ts)
            views.append({
                "cam": cam_name,
                "ts": ts,
                "rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                "keep": (mask == 0).astype(np.float32),  # 1 = background
                "K": K,
                "viewmat": w2c.as_matrix().astype(np.float32),
                "radial": np.array([cam.intr.dist[0], cam.intr.dist[1], cam.intr.dist[4],
                                    0.0, 0.0, 0.0], dtype=np.float32),
            })
    return views


def lidar_init(seq, max_points: int, every: int, device):
    """Aggregate lidar sweeps into the city frame for initialisation."""
    sweeps = sorted((seq.root / "sensors" / "lidar").glob("*.feather"))[::every]
    pts = []
    for f in sweeps:
        d = pd.read_feather(f)
        xyz_ego = d[["x", "y", "z"]].to_numpy().astype(np.float64)
        ego_pose = seq.city_se3_ego(int(f.stem))
        pts.append(ego_pose.apply(xyz_ego))
    xyz = np.concatenate(pts, axis=0)
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[sel]
    return torch.tensor(xyz, dtype=torch.float32, device=device)


def colourise(points, views, device):
    """Give each init point a colour by projecting into the views that see it."""
    N = points.shape[0]
    acc = torch.zeros((N, 3), device=device)
    cnt = torch.zeros((N, 1), device=device)
    pts_np = points.detach().cpu().numpy().astype(np.float64)
    for v in views[::max(1, len(views) // 40)]:
        pose = frames.Pose.from_matrix(v["viewmat"].astype(np.float64))
        cam_pts = pose.apply(pts_np)
        z = cam_pts[:, 2]
        front = z > 0.1
        if front.sum() == 0:
            continue
        K = v["K"]
        u = (K[0, 0] * cam_pts[:, 0] / np.maximum(z, 1e-6) + K[0, 2])
        vv = (K[1, 1] * cam_pts[:, 1] / np.maximum(z, 1e-6) + K[1, 2])
        H, W = v["rgb"].shape[:2]
        ok = front & (u >= 0) & (u < W - 1) & (vv >= 0) & (vv < H - 1)
        if ok.sum() == 0:
            continue
        ui = u[ok].astype(np.int32)
        vi = vv[ok].astype(np.int32)
        keep = v["keep"][vi, ui] > 0  # only colour from background pixels
        idx = np.nonzero(ok)[0][keep]
        if len(idx) == 0:
            continue
        cols = v["rgb"][vi[keep], ui[keep]]
        acc[torch.tensor(idx, device=device)] += torch.tensor(cols, dtype=torch.float32, device=device)
        cnt[torch.tensor(idx, device=device)] += 1
    rgb = torch.where(cnt > 0, acc / cnt.clamp(min=1), torch.full_like(acc, 0.5))
    return rgb


def ssim_torch(a, b):
    """Differentiable SSIM on HWC tensors, used in the training loss."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=torch.float32, device=a.device)
    win = (k @ k.T)[None, None].repeat(3, 1, 1, 1)
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    sa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    sb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return s.mean()


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp(min=1e-10)
    return float(10 * torch.log10(1.0 / mse))


def ssim(a, b):
    """Global SSIM on the whole image (11x11 gaussian window)."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=torch.float32, device=a.device)
    win = (k @ k.T)[None, None].repeat(3, 1, 1, 1)
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    sa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    sb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, default=Path("data/av2_masks"))
    ap.add_argument("--out", type=Path, default=Path("outputs/background_model.pt"))
    ap.add_argument("--cameras", nargs="*", default=list(RING_CAMERAS))
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--init-points", type=int, default=400_000)
    ap.add_argument("--lidar-every", type=int, default=2)
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--cap-max", type=int, default=900_000)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--seg-start", type=float, default=0.0)
    ap.add_argument("--seg-end", type=float, default=1.0)
    ap.add_argument("--seg-margin-m", type=float, default=6.0)
    ap.add_argument("--seg-pad-m", type=float, default=30.0)
    ap.add_argument("--noise-lr", type=float, default=1.6e-4,
                    help="lr fed to MCMC position noise; scaler = this * 5e5")
    ap.add_argument("--means-lr", type=float, default=None,
                    help="if unset, scaled by scene extent (3DGS convention)")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    seq = Av2Sequence(args.seq_dir)
    assert_av2_frame_sane(seq)

    print("loading views ...")
    views = load_views(seq, args.cameras, args.mask_dir, args.downscale, args.stride)
    if args.seg_start > 0.0 or args.seg_end < 1.0:
        # Keep only views whose camera centre falls inside this segment of
        # the ego trajectory (plus margin). Splitting an 83m drive into
        # overlapping chunks is what makes high density affordable.
        ego_ts = seq.image_timestamps("ring_front_center")
        ego = np.stack([seq.city_se3_ego(t).t for t in ego_ts])
        step_d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ego, axis=0), axis=1))])
        total = float(step_d[-1])
        lo = args.seg_start * total - args.seg_margin_m
        hi = args.seg_end * total + args.seg_margin_m
        kept = []
        for v in views:
            c = frames.Pose.from_matrix(v["viewmat"].astype(np.float64)).inverse().apply(
                np.zeros((1, 3)))[0]
            along = float(np.interp(0.0, [0, 1], [0, 1]))  # placeholder, replaced below
            d = np.linalg.norm(ego - c, axis=1)
            along = float(step_d[int(np.argmin(d))])
            if lo <= along <= hi:
                kept.append(v)
        print(f"  segment [{args.seg_start:.2f},{args.seg_end:.2f}] of {total:.1f}m "
              f"-> {len(kept)}/{len(views)} views")
        views = kept
        if not views:
            raise RuntimeError("segment selected zero views")
    train = [v for i, v in enumerate(views) if i % args.holdout_every != 0]
    test = [v for i, v in enumerate(views) if i % args.holdout_every == 0]
    print(f"  {len(views)} views -> {len(train)} train / {len(test)} held out")
    sizes = sorted({(v["rgb"].shape[1], v["rgb"].shape[0]) for v in views})
    print(f"  image sizes present: {sizes}")

    print("initialising from lidar ...")
    means = lidar_init(seq, args.init_points, args.lidar_every, device)
    if args.seg_start > 0.0 or args.seg_end < 1.0:
        cams = np.stack([frames.Pose.from_matrix(v["viewmat"].astype(np.float64))
                         .inverse().apply(np.zeros((1, 3)))[0] for v in views])
        lo_xyz = cams.min(axis=0) - args.seg_pad_m
        hi_xyz = cams.max(axis=0) + args.seg_pad_m
        m = means.detach().cpu().numpy()
        keep = ((m >= lo_xyz) & (m <= hi_xyz)).all(axis=1)
        means = means[torch.tensor(np.nonzero(keep)[0], device=device)]
        print(f"  lidar restricted to segment: {int(keep.sum())} points")
    # AV2 city coordinates sit ~2.8km from the origin, which wastes float32
    # precision and breaks the scale assumptions in the lr schedule. Shift
    # the whole scene (points AND cameras) to its centroid via frames.py.
    origin = means.mean(dim=0)
    shift = frames.Pose(np.eye(3), -origin.detach().cpu().numpy().astype(np.float64))
    means = means - origin
    for v in views:
        w2c = frames.Pose.from_matrix(v["viewmat"].astype(np.float64))
        v["viewmat"] = w2c.compose(shift.inverse()).as_matrix().astype(np.float32)
    print(f"  recentred scene by {origin.detach().cpu().numpy().round(1)}")
    torch.save({"origin": origin.detach().cpu()}, str(args.out) + ".origin.pt")
    N = means.shape[0]
    print(f"  {N} points")
    rgb = colourise(means, train, device)

    # scale init: exact median distance to the 3 nearest neighbours via a
    # CPU KD-tree. A GPU cdist over all points would OOM on 12GB, and
    # subsampling the reference set inflates the distance badly.
    from scipy.spatial import cKDTree

    pts_np = means.detach().cpu().numpy()
    tree = cKDTree(pts_np)
    q = pts_np[np.random.default_rng(0).choice(N, min(N, 50_000), replace=False)]
    dists, _ = tree.query(q, k=4, workers=-1)
    scale0 = float(np.clip(np.median(dists[:, 1:].mean(axis=1)), 0.02, 0.5))
    print(f"  init scale {scale0:.3f} m")

    quats = torch.zeros((N, 4), device=device); quats[:, 0] = 1.0
    sh0 = (rgb - 0.5) / 0.2820948
    shN = torch.zeros((N, (args.sh_degree + 1) ** 2 - 1, 3), device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((N, 3), math.log(scale0), device=device)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(sh0[:, None, :]),
        "shN": torch.nn.Parameter(shN),
    }).to(device)

    # 3DGS scales the means lr by scene extent; this scene is ~80m across,
    # so the stock 1.6e-4 (tuned for unit-scale scenes) cannot move a
    # gaussian far enough to sharpen anything.
    extent = float(torch.quantile(means.norm(dim=1), 0.9))
    means_lr = args.means_lr if args.means_lr is not None else 1.6e-4
    print(f"  scene extent {extent:.1f} m, means lr {means_lr:.2e}, "
          f"mcmc noise lr {args.noise_lr:.2e} (noise scaler {args.noise_lr * 5e5:.0f})")
    lrs = {"means": means_lr, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15)
            for k in params.keys()}

    # MCMC densification: DefaultStrategy needs screen-space means2d
    # gradients, which the UT / eval3d path does not produce.
    strategy = MCMCStrategy(
        cap_max=args.cap_max,
        refine_stop_iter=int(args.iters * 0.75),
        verbose=False,
    )
    state = strategy.initialize_state()

    def render(view, sh_deg):
        # cameras differ in aspect (ring_front_center is portrait), so the
        # render size must come from the view itself, not a global constant.
        H, W = view["rgb"].shape[:2]
        viewmat = torch.tensor(view["viewmat"], device=device)[None]
        K = torch.tensor(view["K"], device=device)[None]
        radial = torch.tensor(view["radial"], device=device)[None]
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        img, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W, height=H,
            sh_degree=sh_deg,
            camera_model="pinhole",
            with_ut=True,
            with_eval3d=True,
            packed=False,  # UT does not support packed mode
            radial_coeffs=radial,
            near_plane=0.2,
            far_plane=250.0,
        )
        return img[0], alpha[0], info

    print(f"training {args.iters} iters ...")
    t0 = time.time()
    rng = np.random.default_rng(0)
    for step in range(args.iters):
        v = train[int(rng.integers(len(train)))]
        sh_deg = min(args.sh_degree, step // (args.iters // (args.sh_degree + 1) + 1))
        img, alpha, info = render(v, sh_deg)
        gt = torch.tensor(v["rgb"], device=device)
        keep = torch.tensor(v["keep"], device=device)[..., None]
        diff = (img - gt).abs() * keep
        l1 = diff.sum() / keep.sum().clamp(min=1) / 3
        # MCMC regularisers (Kheradmand et al.): keep opacities and scales
        # from blowing up as gaussians are relocated.
        dssim = 1.0 - ssim_torch((img * keep).clamp(0, 1), (gt * keep).clamp(0, 1))
        loss = ((1.0 - args.ssim_lambda) * l1
                + args.ssim_lambda * dssim
                + args.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
                + args.scale_reg * torch.exp(params["scales"]).abs().mean())
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, opts, state, step, info, args.noise_lr)
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({k: v.detach().cpu() for k, v in params.items()} |
                       {"sh_degree": args.sh_degree, "iter": step}, args.out)
            print(f"  checkpointed at {step} -> {args.out}", flush=True)
        if step % 500 == 0:
            print(f"  [{step}/{args.iters}] loss={loss.detach().item():.4f} "
                  f"N={params['means'].shape[0]} {time.time()-t0:.0f}s", flush=True)

    print("evaluating on held-out views ...")
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    ps, ss, lp = [], [], []
    with torch.no_grad():
        for v in test:
            img, _, _ = render(v, args.sh_degree)
            gt = torch.tensor(v["rgb"], device=device)
            keep = torch.tensor(v["keep"], device=device)[..., None]
            a = (img * keep).clamp(0, 1)
            b = (gt * keep).clamp(0, 1)
            ps.append(psnr(a, b)); ss.append(ssim(a, b))
            # LPIPS wants NCHW in [-1,1]
            la = (a.permute(2, 0, 1)[None] * 2 - 1)
            lb = (b.permute(2, 0, 1)[None] * 2 - 1)
            lp.append(float(lpips_fn(la, lb).item()))
    metrics = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
               "lpips": float(np.mean(lp)),
               "n_test": len(test), "n_gaussians": int(params["means"].shape[0])}
    print("HELD-OUT:", json.dumps(metrics))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in params.items()} |
               {"sh_degree": args.sh_degree, "metrics": metrics}, args.out)
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
