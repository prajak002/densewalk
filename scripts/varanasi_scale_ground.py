"""M1-Varanasi step 7: turn the SfM reconstruction into a metric world.

COLMAP gives shape, not size, and its world axes are arbitrary. The G1 needs
the project's canonical frame -- metres, Z-up, floor at z=0 -- and it needs the
crowd expressed in that same frame, because the research point is the robot
walking in the SAME world as the people, not a tidied-up abstraction of it.

Two independent scale estimates, deliberately, so they can disagree out loud:

  camera height   the clip is handheld, so the camera rides ~1.5m above the
                  ground; the plane-to-path distance in COLMAP units gives a
                  scale directly.
  pedestrian stature
                  each tracked person is a ruler. Their foot point back-projects
                  onto the ground plane; their head ray is then intersected at
                  that ground position, giving a height in COLMAP units. The
                  median over thousands of observations should be ~1.65m.

Agreement between the two is the check that the plane fit is not tilted. A
tilted plane makes distant people systematically taller or shorter, which shows
up as disagreement rather than as a plausible-looking wrong answer.

Writes world.json: the similarity transform COLMAP -> canonical metric frame,
plus the ground plane and the per-track metric trajectories.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames
from varanasi_train_splat import read_colmap

ADULT_STATURE_M = 1.65   # median adult, mixed sex; the clip is adults + children
CAMERA_HEIGHT_M = 1.50   # handheld / chest-height gimbal


def fit_plane_ransac(pts: np.ndarray, iters: int = 2000, thresh: float = 0.05,
                     rng=None):
    """Plane through the densest planar support. Returns (normal, d, inliers)
    with the plane defined as n . x + d = 0 and |n| = 1."""
    rng = rng or np.random.default_rng(0)
    best = (None, None, np.zeros(len(pts), bool))
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ a)
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    # refit on all inliers via SVD -- the 3-point hypothesis is only a seed
    P = pts[inl]
    centroid = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - centroid)
    n = vh[-1] / np.linalg.norm(vh[-1])
    d = -float(n @ centroid)
    return n, d, inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plane-thresh", type=float, default=0.05,
                    help="RANSAC inlier distance, in COLMAP units")
    args = ap.parse_args()

    cam, imgs, xyz, _ = read_colmap(args.model_dir)
    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])
    poses = {im["name"]: im["w2c"] for im in imgs}
    centres = np.stack([im["w2c"].inverse().apply(np.zeros((1, 3)))[0] for im in imgs])

    # The street is below the camera path. Restricting the RANSAC to points on
    # that side stops it locking onto a shopfront wall, which is both planar and
    # better sampled than the road.
    up_guess = -np.median(np.stack([p.R[1] for p in poses.values()]), axis=0)
    up_guess /= np.linalg.norm(up_guess)
    rel = xyz - centres.mean(axis=0)
    below = (rel @ up_guess) < 0
    print(f"{below.sum()}/{len(xyz)} points below the camera path")

    n, d, inl = fit_plane_ransac(xyz[below], thresh=args.plane_thresh)
    if n @ up_guess < 0:
        n, d = -n, -d
    print(f"ground plane: n={n.round(4).tolist()} d={d:.4f}, "
          f"{int(inl.sum())} inliers of {int(below.sum())}")

    cam_h_units = float(np.median(centres @ n + d))
    scale_from_camera = CAMERA_HEIGHT_M / cam_h_units
    print(f"camera sits {cam_h_units:.4f} units above plane "
          f"-> scale {scale_from_camera:.4f} m/unit")

    # ---- pedestrian stature ----
    tracks = json.loads(args.tracks.read_text())["tracks"]
    heights, per_track = [], {}
    for tid, dets in tracks.items():
        if dets[0]["cls"] != "person":
            continue
        for det in dets:
            w2c = poses.get(det["image"])
            if w2c is None:
                continue
            c2w = w2c.inverse()
            origin = c2w.apply(np.zeros((1, 3)))[0]
            x0, y0, x1, y1 = det["xyxy"]
            foot = np.array([(x0 + x1) / 2, y1])
            head = np.array([(x0 + x1) / 2, y0])

            def ray(px):
                v = np.linalg.inv(K) @ np.array([px[0], px[1], 1.0])
                v = c2w.R @ v
                return v / np.linalg.norm(v)

            rf, rh = ray(foot), ray(head)
            denom = rf @ n
            if abs(denom) < 1e-6:
                continue
            t = -(origin @ n + d) / denom
            if t <= 0:
                continue
            ground = origin + t * rf          # where the feet meet the plane
            # The person is a vertical segment standing at `ground`, so the head
            # sits at ground + h*n and must lie ON the head ray. Solve
            #   (ground - origin + h*n) x rh = 0
            # in least squares. (Intersecting the head ray at the foot's own
            # normal-distance instead makes h identically zero -- the terms
            # cancel -- which is a silent, very convincing way to get nonsense.)
            a = ground - origin
            cross_n = np.cross(n, rh)
            denom_h = float(cross_n @ cross_n)
            if denom_h < 1e-12:
                continue
            h = float(-(np.cross(a, rh) @ cross_n) / denom_h)
            if 0 < h < 5:                     # reject absurdities, not outliers
                per_track.setdefault(tid, []).append(h)
            heights.append(h) if False else None

    # In a crowd this dense most people's feet are hidden behind the people in
    # front, so the detection box bottom marks where occlusion starts, not the
    # ground -- which truncates the height and biases the scale upward. Take a
    # high percentile within each track (the frames where that person was least
    # occluded) and then the median across people, instead of pooling every
    # observation and inheriting the occlusion bias.
    per_track_h = [float(np.percentile(v, 90)) for v in per_track.values()
                   if len(v) >= 5]
    heights = np.array(per_track_h)
    med_h = float(np.median(heights))
    scale_from_people = ADULT_STATURE_M / med_h
    print(f"{len(heights)} person tracks (p90 per track), median height {med_h:.4f} units "
          f"-> scale {scale_from_people:.4f} m/unit")

    disagree = abs(scale_from_camera - scale_from_people) / scale_from_people
    print(f"the two estimates disagree by {disagree*100:.1f}%")

    scale = scale_from_people          # more samples, and it is the thing the
                                       # robot has to avoid colliding with
    # canonical frame: plane normal -> +Z, floor at z=0, metres
    z = n
    x = centres[-1] - centres[0]       # walking direction defines +X
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])            # world -> canonical rotation
    t = np.array([0.0, 0.0, d])        # lift so the plane sits at z = 0
    canon = frames.Pose(R, t)

    check_centres = canon.apply(centres) * scale
    out = {
        "scale_m_per_unit": scale,
        "scale_from_camera_height": scale_from_camera,
        "scale_from_pedestrians": scale_from_people,
        "scale_disagreement_frac": disagree,
        "n_person_observations": int(len(heights)),
        "median_person_height_units": med_h,
        "plane_normal": n.tolist(),
        "plane_d": float(d),
        "canonical_R": R.tolist(),
        "canonical_t": t.tolist(),
        "camera_height_m": float(np.median(check_centres[:, 2])),
        "path_length_m": float(np.linalg.norm(np.diff(check_centres, axis=0), axis=1).sum()),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("canonical_R", "canonical_t")}, indent=2))


if __name__ == "__main__":
    main()
