"""M1-Varanasi step 5: train the static-background splat from COLMAP poses.

Differs from scripts/train_background.py (AV2) in what it is given, not in
what it does:
  poses   SfM instead of rig calibration -- so they carry no metric scale
  init    14,835 sparse SfM points (with COLMAP's own RGB) instead of lidar
  masks   detector-derived instead of cuboid-derived
  views   one forward-walking camera instead of a 7-camera ring

Gaussians are initialised from the sparse points and MCMC densifies up to
--cap-max; with only ~15k seeds the densification is doing most of the work,
unlike the AV2 run which started from 400k lidar points.

Dumps a train-view GT-vs-prediction image early, on purpose: it is the cheap
decisive check that the poses and the loss are wired correctly, and it beats
waiting for a held-out metric at the end of a long run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames


def read_colmap(model_dir: Path):
    """Parse COLMAP's TXT model. Returns (intrinsics dict, list of images)."""
    cam = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        model, w, h = p[1], int(p[2]), int(p[3])
        if model != "SIMPLE_RADIAL":
            raise SystemExit(f"unexpected camera model {model}")
        f, cx, cy, k1 = (float(x) for x in p[4:8])
        cam = {"w": w, "h": h, "f": f, "cx": cx, "cy": cy, "k1": k1}
        break

    imgs = []
    lines = [l for l in (model_dir / "images.txt").read_text().splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        # every coordinate conversion goes through frames.py
        pose = frames.colmap_qvec_to_world_to_cam_pose(
            [float(x) for x in p[1:5]], [float(x) for x in p[5:8]])
        imgs.append({"name": p[9], "w2c": pose})
    imgs.sort(key=lambda d: int(d["name"].split(".")[0]))

    xyz, rgb = [], []
    for line in (model_dir / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return cam, imgs, np.array(xyz), np.array(rgb, dtype=np.float32) / 255.0


def ssim_torch(a, b):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="masks_train: 255 = dynamic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--cap-max", type=int, default=400_000)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--means-lr", type=float, default=1.6e-4)
    ap.add_argument("--noise-lr", type=float, default=1.6e-4,
                    help="fed to MCMC position noise; scaler = this * 5e5. Kept "
                         "separate from means-lr on purpose: coupling them "
                         "inflates the noise and destroys training.")
    ap.add_argument("--check-every", type=int, default=2500)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    cam, imgs, xyz, rgb = read_colmap(args.model_dir)
    print(f"colmap: {len(imgs)} images, {len(xyz)} points, "
          f"{cam['w']}x{cam['h']} f={cam['f']:.1f} k1={cam['k1']:.4f}")

    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]],
                 dtype=np.float32)
    radial = np.array([cam["k1"], 0, 0, 0, 0, 0], dtype=np.float32)

    views = []
    for im in imgs:
        img = cv2.imread(str(args.images / im["name"]))
        if img is None:
            continue
        m = cv2.imread(str(args.masks / f"{im['name']}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros(img.shape[:2], np.uint8)
        views.append({
            "name": im["name"],
            "rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            "keep": (m == 0).astype(np.float32),   # 1 = static background
            "K": K,
            "viewmat": im["w2c"].as_matrix().astype(np.float32),
            "radial": radial,
        })
    train = [v for i, v in enumerate(views) if i % args.holdout_every != 0]
    test = [v for i, v in enumerate(views) if i % args.holdout_every == 0]
    print(f"{len(views)} views -> {len(train)} train / {len(test)} held out")

    # Recentre so float32 precision and the lr schedule behave; COLMAP's
    # origin is the first camera, not the scene centre.
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    origin = means.mean(dim=0)
    shift = frames.Pose(np.eye(3), -origin.detach().cpu().numpy().astype(np.float64))
    means = means - origin
    for v in views:
        w2c = frames.Pose.from_matrix(v["viewmat"].astype(np.float64))
        v["viewmat"] = w2c.compose(shift.inverse()).as_matrix().astype(np.float32)
    torch.save({"origin": origin.detach().cpu()}, str(args.out) + ".origin.pt")

    N = means.shape[0]
    from scipy.spatial import cKDTree
    pts_np = means.detach().cpu().numpy()
    tree = cKDTree(pts_np)
    d, _ = tree.query(pts_np, k=4, workers=-1)
    scale0 = float(np.clip(np.median(d[:, 1:].mean(axis=1)), 1e-3, 1.0))
    extent = float(torch.quantile(means.norm(dim=1), 0.9))
    print(f"{N} seeds, init scale {scale0:.4f}, scene extent {extent:.2f} "
          f"(COLMAP units, NOT metres)")

    quats = torch.zeros((N, 4), device=device); quats[:, 0] = 1.0
    sh0 = (torch.tensor(rgb, dtype=torch.float32, device=device) - 0.5) / 0.2820948
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((N, 3), math.log(scale0), device=device)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(sh0[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((N, (args.sh_degree + 1) ** 2 - 1, 3), device=device)),
    }).to(device)

    lrs = {"means": args.means_lr, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15)
            for k in params}
    strategy = MCMCStrategy(cap_max=args.cap_max,
                            refine_stop_iter=int(args.iters * 0.75), verbose=False)
    state = strategy.initialize_state()

    def render(view, sh_deg):
        H, W = view["rgb"].shape[:2]
        img, alpha, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=torch.tensor(view["viewmat"], device=device)[None],
            Ks=torch.tensor(view["K"], device=device)[None],
            width=W, height=H, sh_degree=sh_deg, camera_model="pinhole",
            with_ut=True, with_eval3d=True, packed=False,
            radial_coeffs=torch.tensor(view["radial"], device=device)[None],
            near_plane=0.01, far_plane=100.0)
        return img[0], alpha[0], info

    def dump_check(step):
        v = train[len(train) // 2]
        with torch.no_grad():
            img, _, _ = render(v, args.sh_degree)
        a = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        b = (v["rgb"] * 255).astype(np.uint8)
        side = np.hstack([cv2.cvtColor(b, cv2.COLOR_RGB2BGR),
                          cv2.cvtColor(a, cv2.COLOR_RGB2BGR)])
        p = args.out.parent / f"trainview_{step:05d}.jpg"
        cv2.imwrite(str(p), side, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  train-view check -> {p}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"training {args.iters} iters ...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    for step in range(args.iters):
        v = train[int(rng.integers(len(train)))]
        sh_deg = min(args.sh_degree, step // (args.iters // (args.sh_degree + 1) + 1))
        img, alpha, info = render(v, sh_deg)
        gt = torch.tensor(v["rgb"], device=device)
        keep = torch.tensor(v["keep"], device=device)[..., None]
        l1 = ((img - gt).abs() * keep).sum() / keep.sum().clamp(min=1) / 3
        dssim = 1.0 - ssim_torch((img * keep).clamp(0, 1), (gt * keep).clamp(0, 1))
        loss = ((1 - args.ssim_lambda) * l1 + args.ssim_lambda * dssim
                + args.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
                + args.scale_reg * torch.exp(params["scales"]).abs().mean())
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, opts, state, step, info, args.noise_lr)
        if step % 500 == 0:
            print(f"  [{step}/{args.iters}] loss={loss.item():.4f} "
                  f"N={params['means'].shape[0]} {time.time()-t0:.0f}s", flush=True)
        if args.check_every and step and step % args.check_every == 0:
            dump_check(step)
    dump_check(args.iters)

    print("evaluating held-out views ...", flush=True)
    ps, ss = [], []
    with torch.no_grad():
        for v in test:
            img, _, _ = render(v, args.sh_degree)
            gt = torch.tensor(v["rgb"], device=device)
            keep = torch.tensor(v["keep"], device=device)[..., None]
            a, b = (img * keep).clamp(0, 1), (gt * keep).clamp(0, 1)
            ps.append(psnr(a, b)); ss.append(float(ssim_torch(a, b)))
    metrics = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
               "n_test": len(test), "n_gaussians": int(params["means"].shape[0]),
               "iters": args.iters, "train_seconds": round(time.time() - t0, 1)}
    print("HELD-OUT: " + json.dumps(metrics), flush=True)
    torch.save({k: p.detach().cpu() for k, p in params.items()} |
               {"sh_degree": args.sh_degree, "metrics": metrics}, args.out)
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
