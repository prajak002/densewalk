"""M1-Varanasi step 8: lift the crowd into the robot's world.

Turns the 82 image-space tracks into metric ground trajectories in the same
canonical frame as the splat and the ground plane -- metres, Z-up, floor at
z=0 -- because the research point is the G1 walking among THESE people, in
THIS street, not among abstract obstacles in a tidied-up copy of it.

Per detection: the foot point (bottom-centre of the box) is back-projected
along its camera ray onto the ground plane. Feet are used rather than box
centres because feet are the one part of a person that provably lies on the
plane we solved; a box centre floats with posture and height.

Each track becomes a moving obstacle with a position, a velocity and a radius.
The radius comes from the detection's own metric width, not a constant, so a
rickshaw is not modelled as a pedestrian.

Timebase: the clip appears speed-ramped (see the crowd-speed check), so the
output carries BOTH the raw timing and a suggested correction factor that puts
median pedestrian speed at --target-speed. Nothing is silently rescaled --
the planner is told, and chooses.
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

FPS = 29.97
# Conservative collision radii (metres) when the measured width is unusable.
FALLBACK_RADIUS = {"person": 0.25, "bicycle": 0.4, "motorcycle": 0.5,
                   "car": 0.9, "truck": 1.2, "bus": 1.4}


def smooth(P: np.ndarray, win: int = 5) -> np.ndarray:
    """Moving average along the trajectory; back-projection jitter at depth is
    large enough that raw finite differences give nonsense velocities."""
    if len(P) < win:
        return P
    k = np.ones(win) / win
    out = np.empty_like(P)
    for a in range(P.shape[1]):
        out[:, a] = np.convolve(P[:, a], k, mode="same")
        out[:win // 2, a] = P[:win // 2, a]
        out[-(win // 2):, a] = P[-(win // 2):, a]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--max-depth-m", type=float, default=25.0,
                    help="beyond this the foot back-projection is too noisy to trust")
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--target-speed", type=float, default=1.20,
                    help="median crowd walking speed used to suggest a timebase fix")
    args = ap.parse_args()

    w = json.loads(args.world.read_text())
    n = np.array(w["plane_normal"])
    d = w["plane_d"]
    scale = w["scale_m_per_unit"]
    canon = frames.Pose(np.array(w["canonical_R"]), np.array(w["canonical_t"]))

    cam, imgs, _, _ = read_colmap(args.model_dir)
    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    poses = {im["name"]: im["w2c"] for im in imgs}
    centres = np.stack([im["w2c"].inverse().apply(np.zeros((1, 3)))[0] for im in imgs])
    cam_path = canon.apply(centres) * scale

    tracks = json.loads(args.tracks.read_text())["tracks"]
    out_tracks, all_speeds = {}, []

    for tid, dets in tracks.items():
        cls = dets[0]["cls"]
        rows = []
        for det in dets:
            w2c = poses.get(det["image"])
            if w2c is None:
                continue
            c2w = w2c.inverse()
            o = c2w.apply(np.zeros((1, 3)))[0]
            x0, y0, x1, y1 = det["xyxy"]
            xm = (x0 + x1) / 2
            r = c2w.R @ (Kinv @ np.array([xm, y1, 1.0]))
            r /= np.linalg.norm(r)
            den = float(r @ n)
            if abs(den) < 1e-6:
                continue
            t = -float(o @ n + d) / den
            if t <= 0 or t * scale > args.max_depth_m:
                continue
            ground = o + t * r

            # metric width: two rays through the box edges, intersected at the
            # same depth, gives the object's footprint width directly
            rl = c2w.R @ (Kinv @ np.array([x0, y1, 1.0])); rl /= np.linalg.norm(rl)
            rr = c2w.R @ (Kinv @ np.array([x1, y1, 1.0])); rr /= np.linalg.norm(rr)
            # signed denominators, exactly as for the ground point above: the plane
            # normal points up, so a downward ray has r.n < 0. Clamping it with
            # max(r.n, 1e-6) (the original code) sent every edge depth to ~1e6 and
            # saturated every radius at the 2.0 m clip.
            dnl, dnr = float(rl @ n), float(rr @ n)
            if abs(dnl) < 1e-6 or abs(dnr) < 1e-6:
                continue
            dl = -float(o @ n + d) / dnl
            dr = -float(o @ n + d) / dnr
            if dl <= 0 or dr <= 0:
                continue
            width_m = float(np.linalg.norm((o + dl * rl) - (o + dr * rr)) * scale)

            rows.append((det["frame"], ground, width_m, t * scale))

        if len(rows) < args.min_obs:
            continue
        rows.sort(key=lambda z: z[0])
        f = np.array([z[0] for z in rows], float)
        P = canon.apply(np.stack([z[1] for z in rows])) * scale
        P = smooth(P)
        widths = np.array([z[2] for z in rows])
        depths = np.array([z[3] for z in rows])

        tsec = f / FPS
        dt = np.gradient(tsec)
        V = np.gradient(P, axis=0) / dt[:, None]
        sp = np.linalg.norm(V[:, :2], axis=1)
        med_sp = float(np.median(sp))
        if cls == "person":
            all_speeds.append(med_sp)

        radius = float(np.clip(np.median(widths) / 2,
                               0.15, 2.0)) if np.isfinite(widths).all() \
            else FALLBACK_RADIUS.get(cls, 0.5)

        out_tracks[tid] = {
            "cls": cls,
            "radius_m": radius,
            "n_obs": len(rows),
            "t_start_s": float(tsec[0]), "t_end_s": float(tsec[-1]),
            "median_speed_mps_raw": med_sp,
            "median_depth_m": float(np.median(depths)),
            # z is dropped: everything stands on the floor at z=0 by construction
            "xy_m": [[float(a), float(b)] for a, b in P[:, :2]],
            "t_s": [float(x) for x in tsec],
            "v_xy_mps_raw": [[float(a), float(b)] for a, b in V[:, :2]],
        }

    med_crowd = float(np.median(all_speeds)) if all_speeds else float("nan")
    time_scale = float(med_crowd / args.target_speed) if med_crowd > 0 else 1.0

    summary = {
        "n_tracks": len(out_tracks),
        "by_class": {c: sum(1 for v in out_tracks.values() if v["cls"] == c)
                     for c in sorted({v["cls"] for v in out_tracks.values()})},
        "median_person_speed_mps_raw": med_crowd,
        "suggested_time_scale": time_scale,
        "suggested_time_scale_note":
            f"raw crowd median is {med_crowd:.2f} m/s; dividing raw speeds by "
            f"{time_scale:.2f} puts it at {args.target_speed:.2f} m/s. Applied "
            f"by the consumer, not here.",
        "frame": "canonical: metres, Z-up, floor z=0",
        "camera_path_m": [[float(a), float(b)] for a, b in cam_path[:, :2]],
    }
    args.out.write_text(json.dumps({"summary": summary, "tracks": out_tracks}, indent=2))
    print(json.dumps(summary | {"camera_path_m": f"<{len(cam_path)} points>"}, indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 9))
        colours = {"person": "#3b7dd8", "motorcycle": "#d84f3b",
                   "bicycle": "#3bd88a", "car": "#9b3bd8", "truck": "#d8a03b"}
        for tid, tr in out_tracks.items():
            P = np.array(tr["xy_m"])
            ax.plot(P[:, 0], P[:, 1], "-", lw=1.1,
                    color=colours.get(tr["cls"], "0.5"), alpha=0.85)
            ax.scatter(P[0, 0], P[0, 1], s=9,
                       color=colours.get(tr["cls"], "0.5"))
        ax.plot(cam_path[:, 0], cam_path[:, 1], "-k", lw=2.5, label="camera / robot path")
        ax.scatter(*cam_path[0, :2], c="g", s=70, zorder=5, label="start")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, walking direction)"); ax.set_ylabel("y (m)")
        ax.set_title(f"Varanasi crowd in the robot's frame — "
                     f"{len(out_tracks)} moving obstacles")
        handles = [plt.Line2D([], [], color=v, label=k) for k, v in colours.items()]
        ax.legend(handles=handles + [plt.Line2D([], [], color="k", lw=2.5,
                                                label="camera path")], loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
