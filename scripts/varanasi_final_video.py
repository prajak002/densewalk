"""Final video: the G1 walking through the REAL Varanasi crowd.

Everything in one frame, all from the same reconstruction:

  street   the 3DGS splat, rendered from the chase camera's pose
  people   the actual people from your video -- RGBA cutouts placed as
           billboards at the metric ground positions their tracks give
  robot    the G1, rendered by Isaac from the same pose, alpha from depth
  overlay  what the robot PERCEIVES: boxes on the people inside its field of
           view that are not occluded, which is the set its planner acted on
  minimap  a top-down strip showing the robot translating down the street

The splat carries no people by design (masking them is what made the static
background clean), so the crowd is re-inserted here as real pixels rather than
as capsules or synthetic avatars.

Draw order is by depth: people farther than the robot, then the robot, then
people nearer than it. A person passing between camera and robot therefore
occludes it, which is what makes the robot look like it is IN the street
rather than pasted on top of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import subprocess

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames

CAM_BACK_M, CAM_UP_M, CAM_PITCH_DEG = 2.6, 1.25, 20.0
FOCAL_MM, H_APERTURE_MM = 18.0, 20.955
VIDEO_FPS = 29.97
STATURE = {"person": 1.65, "bicycle": 1.6, "motorcycle": 1.5,
           "car": 1.5, "truck": 2.6, "bus": 2.9}


class H264Writer:
    """Write frames through ffmpeg/libx264 instead of cv2.VideoWriter.

    OpenCV's mp4v (MPEG-4 Part 2) at its default bitrate destroys this content:
    splat renders are extremely high-frequency, and the result macroblocked into
    unwatchable garbage. Piping raw BGR to x264 at a fixed CRF keeps it clean.
    """

    def __init__(self, path, w, h, fps):
        self.proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def project(P, R_w2c, t_w2c, K):
    """World (canonical metric, COLMAP-shifted) -> pixel. Returns None behind."""
    pc = R_w2c @ P + t_w2c
    if pc[2] <= 0.05:
        return None
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return np.array([u, v]), float(pc[2])


def paste(dst, sprite, cx, y_bot, h_px, depth=None, robot_depth=None,
          robot_alpha=None, only_nearer=False):
    """Alpha-composite a sprite scaled to h_px, centred at cx, feet at y_bot.

    With robot_depth supplied, the sprite is masked per pixel against the
    robot's own depth: `only_nearer` draws only where the person is in FRONT of
    the robot surface, otherwise the person is suppressed wherever the robot is
    nearer. That is what stops the two bodies intersecting.
    """
    if h_px < 6:
        return
    ar = sprite.shape[1] / sprite.shape[0]
    w_px = max(3, int(round(h_px * ar)))
    h_px = int(round(h_px))
    if w_px > 4000 or h_px > 4000:
        return
    s = cv2.resize(sprite, (w_px, h_px), interpolation=cv2.INTER_AREA)
    x0 = int(round(cx - w_px / 2))
    y0 = int(round(y_bot - h_px))
    H, W = dst.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w_px), min(H, y0 + h_px)
    if x1c <= x0c or y1c <= y0c:
        return
    s = s[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    a = (s[..., 3:4].astype(np.float32) / 255.0)
    if robot_depth is not None and depth is not None:
        rd = robot_depth[y0c:y1c, x0c:x1c]
        ra = robot_alpha[y0c:y1c, x0c:x1c]
        on_robot = (ra > 0.35) & np.isfinite(rd)
        person_nearer = on_robot & (depth < rd)
        if only_nearer:
            a = a * person_nearer[..., None].astype(np.float32)
        else:
            a = a * (~(on_robot & ~person_nearer))[..., None].astype(np.float32)
    roi = dst[y0c:y1c, x0c:x1c].astype(np.float32)
    dst[y0c:y1c, x0c:x1c] = (s[..., :3].astype(np.float32) * a
                             + roi * (1 - a)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cap-dir", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--crowd", type=Path, required=True)
    ap.add_argument("--people", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-overlay", action="store_true")
    # overrides for captures made before the cohort scheme was recorded in meta
    ap.add_argument("--cohort-shift", type=float, default=None)
    ap.add_argument("--cohorts", type=int, default=None)
    ap.add_argument("--crowd-period", type=float, default=None)
    ap.add_argument("--max-gaussian-m", type=float, default=5.0,
                    help="drop Gaussians longer than this. Pruning hard (0.6m, "
                         "73%% of them) made things WORSE: the large Gaussians "
                         "are the background fill, and without them the scene "
                         "gains black holes while the remaining ones still "
                         "streak. The streaking is a viewpoint problem, not a "
                         "size problem, so this is left nearly off.")
    ap.add_argument("--min-opacity", type=float, default=0.004,
                    help="71%% of Gaussians sit below 0.05 opacity and only "
                         "add haze")
    args = ap.parse_args()

    device = "cuda"
    w = json.loads(args.world.read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    meta = json.loads((args.cap_dir / "meta.json").read_text())
    start_xy = np.array(meta["start_xy"])
    yaw0 = float(meta["spawn_yaw"])
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    S2W = np.array([[c0, s0], [-s0, c0]])
    W, H = meta["width"], meta["height"]
    cam_back = float(meta.get("cam_back", CAM_BACK_M))
    cam_up = float(meta.get("cam_up", CAM_UP_M))
    cam_pitch = float(meta.get("cam_pitch_deg", CAM_PITCH_DEG))

    crowd = json.loads(args.crowd.read_text())
    time_scale = crowd["summary"]["suggested_time_scale"]
    ctracks = crowd["tracks"]

    people = json.loads((args.people / "sprites.json").read_text())["tracks"]
    sprite_root = args.people / "sprites"
    # The sprites came from a second tracking pass with its own ids, so metric
    # track 7 and sprite track 7 are different people. Without this map every
    # lookup missed and the street rendered empty.
    id_map = json.loads((args.people / "id_map.json").read_text())
    print(f"{len(people)} sprite tracks, {len(ctracks)} metric tracks, "
          f"{len(id_map)} matched")

    ck = torch.load(args.ckpt, map_location=device)
    origin = torch.load(str(args.ckpt) + ".origin.pt")["origin"].numpy().astype(np.float64)
    sh_degree = int(ck["sh_degree"])
    params = {k: ck[k].to(device) for k in
              ("means", "scales", "quats", "opacities", "sh0", "shN")}

    # Prune the streak-makers before rendering.
    max_axis_units = args.max_gaussian_m / scale
    keep = (torch.exp(params["scales"]).max(dim=1).values < max_axis_units) \
        & (torch.sigmoid(params["opacities"]) > args.min_opacity)
    n0 = params["means"].shape[0]
    params = {k: v[keep] for k, v in params.items()}
    print(f"gaussians {n0} -> {params['means'].shape[0]} "
          f"(dropped {100 * (1 - params['means'].shape[0] / n0):.1f}%: "
          f">{args.max_gaussian_m}m or opacity<{args.min_opacity})")
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)

    fx = W * FOCAL_MM / H_APERTURE_MM
    Kc = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    K = torch.tensor(Kc, dtype=torch.float32, device=device)[None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = H264Writer(args.out, W, H, args.fps)
    pitch = np.deg2rad(cam_pitch)
    sprite_cache: dict[str, np.ndarray] = {}

    # minimap extent from the whole recorded crowd
    allP = np.concatenate([np.array(t["xy_m"]) for t in ctracks.values()])
    # Replay the crowd exactly as the simulation did. Deriving these here
    # instead left the video drawing one distant cohort while the sim ran five
    # overlapping ones: the HUD said 17 people and the street looked empty.
    cohort_shift = float(args.cohort_shift if args.cohort_shift is not None
                         else meta.get("cohort_shift", 12.0))
    n_cohorts = int(args.cohorts if args.cohorts is not None
                    else meta.get("cohorts", 3))
    crowd_period = float(args.crowd_period if args.crowd_period is not None
                         else meta.get("crowd_period", 17.4))
    print(f"replaying {n_cohorts} cohorts, {cohort_shift:.1f}m apart, "
          f"period {crowd_period:.1f}s")
    mm_x = (allP[:, 0].min() - 2, allP[:, 0].max() + 2)
    mm_y = (allP[:, 1].min() - 1, allP[:, 1].max() + 1)
    MM_W, MM_H = 360, 132

    n = 0
    with torch.no_grad():
        for fr in meta["frames"]:
            layer = cv2.imread(str(args.cap_dir / "robot" / f"{fr['i']:05d}.png"),
                               cv2.IMREAD_UNCHANGED)
            if layer is None:
                continue
            t = fr["t"]
            tq = fr.get("t_crowd", t)        # cohort-local crowd time
            cohort = int(fr.get("cohort", 0)) # which replay of the crowd

            xy = S2W @ np.array(fr["robot_xy_sim"]) + start_xy
            yaw = fr["yaw_sim"] + yaw0
            fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            cam_pos = np.array([xy[0], xy[1], fr["robot_z"]]) \
                - cam_back * fwd + np.array([0, 0, cam_up])
            look = np.array([np.cos(yaw) * np.cos(pitch),
                             np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(look, up); right /= np.linalg.norm(right)
            down = np.cross(look, right)
            R_c2w_canon = np.stack([right, down, look], axis=1)

            p_colmap = canon.inverse().apply((cam_pos / scale)[None])[0] - origin
            Rw = canon.R.T @ R_c2w_canon
            R_w2c = Rw.T
            t_w2c = -R_w2c @ p_colmap
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
                packed=False, near_plane=0.01, far_plane=100.0)
            frame = cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255)
                                 .astype(np.uint8), cv2.COLOR_RGB2BGR)

            # ---- place the real people as billboards ----
            drawn_missing = []
            vis_set = {(round(float((S2W @ np.asarray(v) + start_xy)[0]), 2),
                        round(float((S2W @ np.asarray(v) + start_xy)[1]), 2))
                       for v in fr["visible_xy"]}
            drawn = []
            half = n_cohorts // 2
            for ck in range(cohort - half, cohort + half + 1):
                tqk = t - ck * crowd_period
                if tqk < 0:
                    continue
                for tid, tr in ctracks.items():
                    ts_arr = np.array(tr["t_s"]) * time_scale
                    if tqk < ts_arr[0] or tqk > ts_arr[-1]:
                        continue
                    px = float(np.interp(tqk, ts_arr, np.array(tr["xy_m"])[:, 0])) \
                        + ck * cohort_shift
                    py = float(np.interp(tqk, ts_arr, np.array(tr["xy_m"])[:, 1]))
                    ground = np.array([px, py, 0.0])
                    gc = canon.inverse().apply((ground / scale)[None])[0] - origin
                    pr = project(gc, R_w2c, t_w2c, Kc)
                    if pr is None:
                        continue
                    uv_b, depth_u = pr
                    # project() returns depth in COLMAP units, not metres. Labelling
                    # it as metres understated every distance by the scale factor
                    # and, worse, compared it against the robot's 2.6 m for the
                    # draw order, so the near/far split around the robot was wrong.
                    depth = depth_u * scale
                    hgt = STATURE.get(tr["cls"], 1.6)
                    top = np.array([px, py, hgt])
                    tc = canon.inverse().apply((top / scale)[None])[0] - origin
                    pr_t = project(tc, R_w2c, t_w2c, Kc)
                    if pr_t is None:
                        continue
                    uv_t, _ = pr_t
                    h_px = float(abs(uv_b[1] - uv_t[1]))
                    if h_px < 8 or uv_b[0] < -400 or uv_b[0] > W + 400:
                        continue

                    mapped = id_map.get(tid)
                    sp = people.get(mapped["sprite_id"]) if mapped else None
                    if not sp:
                        # No sprite matched this track. Still mark the person so the
                        # scene does not silently lose people the robot is avoiding.
                        drawn_missing.append((uv_b[0], uv_b[1], h_px, tr["cls"],
                                              (round(px, 2), round(py, 2)) in vis_set))
                        continue
                    vframe = int(round(tqk / time_scale * VIDEO_FPS))
                    best = min(sp, key=lambda d: abs(d["frame"] - vframe))
                    if abs(best["frame"] - vframe) > 30:
                        continue
                    key = best["file"]
                    if key not in sprite_cache:
                        im = cv2.imread(str(sprite_root / key), cv2.IMREAD_UNCHANGED)
                        if im is None or im.shape[2] != 4:
                            continue
                        sprite_cache[key] = im
                    drawn.append((depth, sprite_cache[key], uv_b[0], uv_b[1], h_px,
                                  (round(px, 2), round(py, 2)) in vis_set, tr["cls"]))

                # Composite by DEPTH, per pixel. Splitting the crowd into "in
                # front of" and "behind" the robot at one assumed depth made bodies
                # intersect: part of a person can be nearer than the robot's torso
                # while the rest is further, and a flat split cannot express that.
            drawn.sort(key=lambda z: -z[0])       # far to near
            rob_depth = None
            dpath = args.cap_dir / "robot" / f"{fr['i']:05d}_depth.png"
            if dpath.exists():
                dmm = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
                if dmm is not None:
                    rob_depth = dmm.astype(np.float32) / 1000.0
                    rob_depth[rob_depth <= 0] = np.inf

            a = (layer[..., 3:4].astype(np.float32) / 255.0)
            rob = layer[..., :3].astype(np.float32)
            rob[..., 1] = np.minimum(rob[..., 1], np.maximum(rob[..., 0], rob[..., 2]))
            am = cv2.erode((a[..., 0] * 255).astype(np.uint8), np.ones((3, 3), np.uint8))
            a = cv2.GaussianBlur(am.astype(np.float32) / 255.0, (3, 3), 0)[..., None]

            if rob_depth is None:
                for d in drawn:
                    paste(frame, d[1], d[2], d[3], d[4])
                frame = (rob * a + frame.astype(np.float32) * (1 - a)).astype(np.uint8)
            else:
                # people further than the robot's surface, then the robot,
                # then people nearer -- all evaluated per pixel
                for d in drawn:
                    paste(frame, d[1], d[2], d[3], d[4], depth=d[0],
                          robot_depth=rob_depth, robot_alpha=a[..., 0])
                frame = (rob * a + frame.astype(np.float32) * (1 - a)).astype(np.uint8)
                # anyone nearer than the robot must be drawn over it
                for d in drawn:
                    paste(frame, d[1], d[2], d[3], d[4], depth=d[0],
                          robot_depth=rob_depth, robot_alpha=a[..., 0],
                          only_nearer=True)

            # ---- what the robot perceives ----
            if not args.no_overlay:
                for depth, spr, cx, yb, hp, seen, cls in drawn:
                    if not seen:
                        continue
                    wpx = hp * spr.shape[1] / spr.shape[0]
                    p0 = (int(cx - wpx / 2), int(yb - hp))
                    p1 = (int(cx + wpx / 2), int(yb))
                    cv2.rectangle(frame, p0, p1, (80, 230, 80), 2)
                    cv2.putText(frame, f"{cls} {depth:.1f}m",
                                (p0[0], max(12, p0[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 230, 80), 1,
                                cv2.LINE_AA)

            # ---- minimap: shows the robot actually translating ----
            mm = np.full((MM_H, MM_W, 3), 24, np.uint8)

            def to_mm(p):
                u = int((p[0] - mm_x[0]) / (mm_x[1] - mm_x[0]) * (MM_W - 8)) + 4
                v = int((p[1] - mm_y[0]) / (mm_y[1] - mm_y[0]) * (MM_H - 8)) + 4
                return u, MM_H - v
            # logged positions are in the SIM frame; the minimap extents come
            # from the canonical metric crowd, so convert before plotting
            def sim_to_canon(q):
                return S2W @ np.asarray(q) + start_xy
            for q in fr["all_xy"]:
                cv2.circle(mm, to_mm(sim_to_canon(q)), 2, (150, 150, 150), -1)
            for q in fr["visible_xy"]:
                cv2.circle(mm, to_mm(sim_to_canon(q)), 3, (80, 230, 80), -1)
            cv2.circle(mm, to_mm(xy), 4, (60, 90, 240), -1)
            hd = to_mm(xy + 1.6 * np.array([np.cos(yaw), np.sin(yaw)]))
            cv2.line(mm, to_mm(xy), hd, (60, 90, 240), 2)
            frame[10:10 + MM_H, W - MM_W - 10:W - 10] = mm
            cv2.rectangle(frame, (W - MM_W - 10, 10), (W - 10, 10 + MM_H),
                          (200, 200, 200), 1)

            cv2.rectangle(frame, (0, H - 46), (W, H), (0, 0, 0), -1)
            cv2.putText(frame, f"t={t:5.1f}s   goal {fr['goal_dist_m']:5.1f} m   "
                               f"crowd {fr['n_obstacles']:2d}   "
                               f"perceived {fr['n_visible']:2d}   "
                               f"occluded {fr['n_occluded']:2d}"
                               + ("   HOLDING" if fr["stopped"] else ""),
                        (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(frame)
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames (sprite cache {len(sprite_cache)})", flush=True)
    vw.release()
    print(f"wrote {args.out} ({n} frames)")


main()
