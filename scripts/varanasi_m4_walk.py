"""M4: the G1 walks the reconstructed Varanasi street among its real crowd.

One world, per the research point: the robot's start, its goal, the ground
plane and all 69 moving obstacles come from the SAME metric reconstruction
(world.json + crowd_world.json) as the splat. Nothing here is a tidied-up
stand-in for the street -- the people are the people who were actually there,
moving on the trajectories they actually took.

Division of labour:
  locomotion   outputs/g1_policy/model_1499.pt (rsl_rl, 1500 iters, reward
               28.09). It was trained on flat ground with a commanded
               velocity and has never seen an obstacle. It walks; it does not
               decide where.
  navigation   densewalk.crowd_nav decides the command from the crowd state.

Splats are visual only and are not present in the physics scene at all; the
robot stands on Isaac Lab's ground plane at z=0, per the project invariants.

Crowd obstacles are evaluated analytically rather than spawned as kinematic
colliders: teleporting rigid bodies into a humanoid injects unbounded contact
forces and the resulting faceplant tells you nothing about the navigation.
Clearance and collisions are measured from the same geometry the planner sees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--crowd", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--plot", default=None)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--time-scale", type=float, default=None,
                    help="divide raw crowd speeds by this; default uses the "
                         "suggested_time_scale recorded in crowd_world.json")
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True}).app  # noqa: E402

import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers the task ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk.crowd_nav import NavParams, Obstacle, command  # noqa: E402


def build_actor(ckpt_path: str, device: str) -> nn.Module:
    """Rebuild the rsl_rl actor directly.

    Shapes and activation are read off the checkpoint and the installed G1
    config (hidden_dims [256,128,128], activation elu) rather than assumed --
    a silent architecture mismatch would load, run, and walk like it is drunk.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["actor_state_dict"]
    dims = [sd["mlp.0.weight"].shape[1]]
    for i in (0, 2, 4, 6):
        dims.append(sd[f"mlp.{i}.weight"].shape[0])
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers)
    mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items()
                         if k.startswith("mlp.")})
    print(f"actor rebuilt: {dims}, iter {ck.get('iter')}")
    return mlp.to(device).eval()


class Crowd:
    """The recorded crowd, queryable at any wall-clock time."""

    def __init__(self, path: str, time_scale: float):
        blob = json.loads(Path(path).read_text())
        self.summary = blob["summary"]
        self.time_scale = time_scale
        self.tracks = []
        for tid, tr in blob["tracks"].items():
            t = np.array(tr["t_s"]) * time_scale      # stretch out the ramp
            P = np.array(tr["xy_m"])
            V = np.array(tr["v_xy_mps_raw"]) / time_scale
            self.tracks.append({"id": tid, "cls": tr["cls"],
                                "r": tr["radius_m"], "t": t, "P": P, "V": V})

    def at(self, t: float) -> list[Obstacle]:
        out = []
        for tr in self.tracks:
            if t < tr["t"][0] or t > tr["t"][-1]:
                continue                               # not in frame yet / gone
            x = np.interp(t, tr["t"], tr["P"][:, 0])
            y = np.interp(t, tr["t"], tr["P"][:, 1])
            vx = np.interp(t, tr["t"], tr["V"][:, 0])
            vy = np.interp(t, tr["t"], tr["V"][:, 1])
            out.append(Obstacle(np.array([x, y]), np.array([vx, vy]), tr["r"]))
        return out


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    device = "cuda:0"
    cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    # The velocity task times out at 20s and resets the robot to the origin,
    # which silently teleported it backwards mid-traverse. This is one
    # continuous walk down one street, so the episode must outlast the run.
    cfg.episode_length_s = args_cli.seconds + 10.0
    # The velocity term runs its OWN heading controller and resamples a random
    # heading every 10s, overwriting the yaw-rate channel each step. With it on,
    # navigation's turn commands were silently discarded and the robot curved
    # off on the env's random target -- which is why every earlier run wandered
    # and made no progress. Here the command comes from crowd_nav, so both the
    # heading controller and the resampling must be off.
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()

    # The velocity env randomises the robot's initial yaw. Overriding the root
    # pose to fix that made the G1 collapse on the next step, terminate, and
    # reset straight back to a random yaw -- so instead of fighting the env,
    # rotate the WORLD to match whatever heading it gave us. The street is then
    # aligned with the robot's initial facing by construction, and no pose is
    # written. sim = R(yaw0) @ (world - start).
    robot = u.scene["robot"]
    yaw0 = yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    W2S = np.array([[c0, -s0], [s0, c0]])      # world -> sim rotation
    print(f"spawn yaw {yaw0:+.3f} rad; rotating the street to match it")

    actor = build_actor(args_cli.policy, device)

    crowd_blob = json.loads(Path(args_cli.crowd).read_text())
    ts = args_cli.time_scale or crowd_blob["summary"]["suggested_time_scale"]
    crowd = Crowd(args_cli.crowd, ts)
    print(f"crowd: {len(crowd.tracks)} tracks, time_scale {ts:.3f}")

    # The robot starts where the camera did and heads where the camera went:
    # that is the same traversal of the same street, which is the comparison
    # the research question wants.
    cam = np.array(crowd_blob["summary"]["camera_path_m"])
    start_xy, goal_world = cam[0], cam[-1]

    def to_sim(xy):
        return (W2S @ (np.asarray(xy) - start_xy).T).T

    goal_xy = to_sim(goal_world)
    print(f"goal in sim frame {goal_xy.round(2).tolist()} "
          f"({np.linalg.norm(goal_xy):.1f} m away)")

    # The street corridor: the people walking it define where it is. This is
    # more trustworthy than the sparse road points (which failed the ground
    # plane fit -- the road is barely reconstructed because the crowd covers
    # it), and it comes from the same world, not from an invented box.
    all_y = np.concatenate([to_sim(tr["P"])[:, 1] for tr in crowd.tracks])
    corridor = (float(np.percentile(all_y, 5)), float(np.percentile(all_y, 95)))
    print(f"street corridor from crowd extent: y in "
          f"[{corridor[0]:.2f}, {corridor[1]:.2f}] m "
          f"({corridor[1] - corridor[0]:.1f} m wide)")

    dt = float(u.step_dt)
    n_steps = int(args_cli.seconds / dt)
    cmd_term = u.command_manager._terms["base_velocity"]
    p = NavParams()

    log = []
    actions = torch.zeros((1, 37), device=device)
    for step in range(n_steps):
        t = step * dt
        root = u.scene["robot"].data.root_pos_w[0].cpu().numpy()
        quat = u.scene["robot"].data.root_quat_w[0].cpu().numpy()
        # sim origin is the start point of the recorded traversal
        pos_xy = np.array([root[0], root[1]])       # already the sim frame
        yaw = yaw_from_quat(quat)
        vel_w = u.scene["robot"].data.root_lin_vel_w[0].cpu().numpy()[:2]

        obstacles = [Obstacle(to_sim(ob.pos), W2S @ ob.vel, ob.radius)
                     for ob in crowd.at(t)]
        cmd, info = command(pos_xy, yaw, vel_w, goal_xy, obstacles, p,
                            corridor=corridor)

        cmd_term.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32,
                                                 device=device)

        with torch.no_grad():
            flat = obs["policy"] if isinstance(obs, dict) else obs
            actions = actor(flat)
        obs, _, terminated, truncated, _ = env.step(actions)

        clearance = np.inf
        for ob in obstacles:
            clearance = min(clearance, float(np.linalg.norm(ob.pos - pos_xy)
                                             - ob.radius - p.robot_radius))
        log.append({"t": t, "x": float(pos_xy[0]), "y": float(pos_xy[1]),
                    "z": float(root[2]), "yaw": yaw,
                    "cmd": [float(c) for c in cmd],
                    "n_obstacles": len(obstacles),
                    "clearance_m": float(clearance) if np.isfinite(clearance) else None,
                    "min_ttc": info["min_ttc"] if np.isfinite(info["min_ttc"]) else None,
                    "stopped": bool(info["stopped"]),
                    "wall_push": float(info.get("wall_push", 0.0)),
                    "goal_dist_m": float(info["goal_dist"])})

        if bool(terminated[0]) or bool(truncated[0]):
            print(f"episode ended at step {step} (t={t:.2f}s) -- robot fell or timed out")
            obs, _ = env.reset()

        if step % 200 == 0:
            print(f"  [{step}/{n_steps}] t={t:5.2f}s goal_dist={info['goal_dist']:5.1f}m "
                  f"obstacles={len(obstacles):2d} clearance="
                  f"{clearance if np.isfinite(clearance) else float('nan'):5.2f}m "
                  f"z={root[2]:.2f}", flush=True)

    L = [r for r in log if r["clearance_m"] is not None]
    clears = np.array([r["clearance_m"] for r in L]) if L else np.array([np.nan])
    heights = np.array([r["z"] for r in log])
    summary = {
        "steps": len(log), "dt": dt, "seconds": len(log) * dt,
        "time_scale": ts,
        "start_xy": start_xy.tolist(), "goal_xy": goal_xy.tolist(),
        "spawn_yaw": float(yaw0),
        "final_goal_dist_m": log[-1]["goal_dist_m"],
        "distance_travelled_m": float(np.sum(np.linalg.norm(
            np.diff(np.array([[r["x"], r["y"]] for r in log]), axis=0), axis=1))),
        "min_clearance_m": float(np.nanmin(clears)),
        "median_clearance_m": float(np.nanmedian(clears)),
        "collisions": int((clears < 0).sum()),
        "steps_stopped": int(sum(r["stopped"] for r in log)),
        "corridor_y": list(corridor),
        "steps_touching_wall": int(sum(r["wall_push"] > 0 for r in log)),
        "max_abs_y_m": float(np.max(np.abs([r["y"] for r in log]))),
        "min_base_height_m": float(heights.min()),
        "fell": bool(heights.min() < 0.4),
    }
    Path(args_cli.out).write_text(json.dumps({"summary": summary, "log": log}, indent=2))
    print("M4_SUMMARY " + json.dumps(summary))

    if args_cli.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = np.array([[r["x"], r["y"]] for r in log])
        fig, ax = plt.subplots(figsize=(12, 7))
        for tr in crowd.tracks:
            Q = to_sim(tr["P"])
            ax.plot(Q[:, 0], Q[:, 1], "-", lw=0.8, color="0.75")
        C = to_sim(cam)
        ax.plot(C[:, 0], C[:, 1], "--", color="0.35", lw=1.6, label="human path (camera)")
        ax.axhline(corridor[0], color="#8a6d3b", lw=1.2, ls=":", label="street corridor")
        ax.axhline(corridor[1], color="#8a6d3b", lw=1.2, ls=":")
        ax.plot(P[:, 0], P[:, 1], "-", color="#d1452b", lw=2.6, label="G1 path")
        ax.scatter(0, 0, c="g", s=80, zorder=6, label="start")
        ax.scatter(*goal_xy, c="b", s=80, zorder=6, label="goal")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"G1 through the Varanasi crowd — min clearance "
                     f"{summary['min_clearance_m']:.2f} m, "
                     f"{summary['collisions']} collisions")
        ax.legend()
        plt.tight_layout(); plt.savefig(args_cli.plot, dpi=110)
        print(f"wrote {args_cli.plot}")

    env.close()
    app.close()


main()
