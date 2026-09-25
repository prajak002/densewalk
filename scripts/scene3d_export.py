"""Export the Varanasi M4 world for the three.js scene viewer (outputs/scene3d/).

Everything lands in ONE frame: canonical metric (metres, Z-up, floor z=0), the
same frame crowd_world.json and world.json use. Sources:

  static scene  COLMAP sparse points (outputs/varanasi/sparse/*.bin), mapped
                COLMAP -> metric by world.json: p_m = scale * canon(p_colmap)
  camera path   the phone's solved poses, same mapping (checked against
                crowd_world.json camera_path_m and the solved 1.62 m height)
  crowd         crowd_world.json tracks, time stretched by time_scale exactly
                as varanasi_m4_walk.Crowd does (t = t_s * time_scale)
  robot         m4_run8.json, which is logged in the SIM frame
                sim = R(yaw0) @ (world - start); inverted here

The COLMAP .bin readers follow COLMAP's documented binary layout; the Mac copy
of the model has no TXT export for varanasi_train_splat.read_colmap to parse.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk import frames  # noqa: E402


def read_points3d_bin(path: Path):
    xyz, rgb, err = [], [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _pid, x, y, z, r, g, b, e = struct.unpack("<QdddBBBd", f.read(43))
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)                      # (image_id, point2D_idx) int32 pairs
            xyz.append((x, y, z)); rgb.append((r, g, b)); err.append(e)
    return np.array(xyz), np.array(rgb, dtype=np.uint8), np.array(err)


def read_images_bin(path: Path):
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _iid, qw, qx, qy, qz, tx, ty, tz, _cid = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2d)                           # (x, y double, point3D_id int64)
            w2c = frames.colmap_qvec_to_world_to_cam_pose([qw, qx, qy, qz], [tx, ty, tz])
            out.append({"name": name.decode(), "w2c": w2c})
    out.sort(key=lambda d: int(d["name"].split(".")[0]))
    return out


MAX_SPEED = 3.0      # m/s after the time-scale fix; faster = depth/ID jump, not walking
MIN_TRACK_S = 0.5
SMOOTH_S = 0.35      # gaussian sigma for the ground trajectory


def smooth_track(t, xy):
    """Remove tracking jumps and jitter from a ground trajectory.

    Samples implying > MAX_SPEED from a running median are replaced by that
    median, then the path is gaussian-smoothed in time. Returns (xy, n_replaced)
    or (None, 0) for tracks too short to trust.
    """
    if len(t) < 3 or t[-1] - t[0] < MIN_TRACK_S:
        return None, 0
    k = 7
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    med = np.stack([np.median(pad[i:i + k], axis=0) for i in range(len(xy))])
    dt = max(np.median(np.diff(t)), 1e-3)
    bad = np.linalg.norm(xy - med, axis=1) > MAX_SPEED * dt * (k // 2)
    xy = np.where(bad[:, None], med, xy)
    w = np.exp(-0.5 * ((t[:, None] - t[None, :]) / SMOOTH_S) ** 2)
    xy = (w @ xy) / w.sum(1, keepdims=True)
    return xy, int(bad.sum())


def r(a, n=3):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", default="outputs/varanasi")
    ap.add_argument("--run", default="outputs/varanasi/m4_run8.json")
    ap.add_argument("--out", default="outputs/scene3d/scene.json")
    ap.add_argument("--max-err", type=float, default=2.0, help="drop SfM points with reprojection error above this (px)")
    args = ap.parse_args()
    V = Path(args.vdir)

    w = json.loads((V / "world.json").read_text())
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    def to_metric(p_colmap):
        return canon.apply(p_colmap) * scale

    # --- static scene: SfM points ---
    xyz, rgb, err = read_points3d_bin(V / "sparse" / "points3D.bin")
    keep = err <= args.max_err
    pts = to_metric(xyz[keep])
    col = rgb[keep]
    # the street is ~60 m long; drop far SfM outliers (sky/triangulation noise)
    ctr = np.median(pts, axis=0)
    near = np.linalg.norm(pts[:, :2] - ctr[:2], axis=1) < 60
    pts, col = pts[near], col[near]
    z_floor = float(np.percentile(pts[:, 2], 2))

    # --- camera path ---
    imgs = read_images_bin(V / "sparse" / "images.bin")
    cam_c = np.array([to_metric(im["w2c"].inverse().t[None])[0] for im in imgs])
    # camera forward (+z of the OpenCV camera) in metric frame, for the frustum
    cam_fwd = np.array([canon.R @ (im["w2c"].inverse().R @ np.array([0, 0, 1.0])) for im in imgs])

    crowd = json.loads((V / "crowd_world.json").read_text())
    cp = np.array(crowd["summary"]["camera_path_m"])
    xy_err = np.linalg.norm(cam_c[:, :2] - cp, axis=1)
    print(f"camera path vs crowd_world: median {np.median(xy_err):.3f} m, max {xy_err.max():.3f} m")
    print(f"camera height: median {np.median(cam_c[:, 2]):.3f} m (world.json says {w['camera_height_m']:.3f})")
    print(f"SfM points kept {len(pts)} / {len(xyz)}; 2nd-percentile z {z_floor:+.3f} m")
    assert np.median(xy_err) < 0.05, "camera path does not match crowd_world -- frame mismatch"
    assert abs(np.median(cam_c[:, 2]) - w["camera_height_m"]) < 0.1, "camera height mismatch"

    # --- crowd, time-stretched like varanasi_m4_walk.Crowd ---
    run = json.loads(Path(args.run).read_text())
    ts = run["summary"]["time_scale"]
    tracks, n_dropped, n_fixed = [], 0, 0
    for tid, tr in crowd["tracks"].items():
        t_sc = np.array(tr["t_s"]) * ts
        xy_s, n_fix = smooth_track(t_sc, np.array(tr["xy_m"], dtype=np.float64))
        if xy_s is None:
            n_dropped += 1
            continue
        n_fixed += n_fix
        tracks.append({"id": tid, "cls": tr["cls"], "t": r(t_sc, 3), "xy": r(xy_s, 3)})
    print(f"tracks: kept {len(tracks)}, dropped {n_dropped} (< {MIN_TRACK_S} s), "
          f"{n_fixed} jump samples replaced (> {MAX_SPEED} m/s)")

    # --- robot: sim frame -> world. sim = R(yaw0) @ (world - start) ---
    s = run["summary"]
    yaw0 = s["spawn_yaw"]
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    S2W = np.array([[c0, -s0], [s0, c0]]).T
    start = np.array(s["start_xy"])
    log = run["log"]
    rob_xy = np.array([[e["x"], e["y"]] for e in log]) @ S2W.T + start
    rob = {"t": r([e["t"] for e in log], 3), "xy": r(rob_xy, 3),
           "z": r([e["z"] for e in log], 3),
           "yaw": r([e["yaw"] - yaw0 for e in log], 4),
           "goal_dist": r([e["goal_dist_m"] for e in log], 2),
           "clearance": [None if e["clearance_m"] is None else round(e["clearance_m"], 2) for e in log],
           "n_obs": [e["n_obstacles"] for e in log],
           "stopped": [int(e["stopped"]) for e in log]}
    goal_world = S2W @ np.array(s["goal_xy"]) + start
    # corridor edges are horizontal lines in the sim frame; map both ends of each
    corridor = [r(np.array([[x, yc] for x in (-40.0, 40.0)]) @ S2W.T + start) for yc in s["corridor_y"]]
    print(f"robot start {r(rob_xy[0],2)} goal {r(goal_world,2)} (camera path end {r(cp[-1],2)})")

    out = {
        "frame": "canonical metric: metres, Z-up, floor z=0",
        "time_scale": ts,
        "video_fps": 29.97,
        "points": {"xyz": r(pts, 3), "rgb": col.tolist()},
        "camera": {"xyz": r(cam_c, 3), "fwd": r(cam_fwd, 3)},
        "corridor": corridor,
        "tracks": tracks,
        "robot": rob,
        "goal": r(goal_world, 3),
        "summary": {k: s[k] for k in ("collisions", "min_clearance_m", "final_goal_dist_m", "distance_travelled_m")},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
