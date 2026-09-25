"""M4 video: record the G1 walking the Varanasi street through the crowd.

Same navigation as varanasi_m4_walk.py -- this adds what you can watch:
the 69 recorded people and vehicles drawn as capsules at their real metric
positions, and a chase camera following the robot down the street.

The crowd markers are VISUAL ONLY. They carry no collider, deliberately:
teleporting kinematic bodies into a humanoid injects unbounded contact forces
and the resulting faceplant says nothing about the navigation. Clearance is
measured analytically from the same geometry the planner uses, exactly as in
the headless run, so the numbers here match that run's numbers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--crowd", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--seconds", type=float, default=40.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--frame-stride", type=int, default=2)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
parser.add_argument("--cam", default="chase", choices=["chase", "side", "top"])
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True}).app  # noqa: E402

import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk.crowd_nav import NavParams, Obstacle, command  # noqa: E402

MAX_MARKERS = 90          # more than the 69 tracks, fixed so the buffer is static


def build_actor(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["actor_state_dict"]
    dims = [sd["mlp.0.weight"].shape[1]] + [sd[f"mlp.{i}.weight"].shape[0]
                                            for i in (0, 2, 4, 6)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    return mlp.to(device).eval()


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    device = "cuda:0"
    cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    cfg.episode_length_s = args_cli.seconds + 10.0
    # see notes in varanasi_m4_walk.py: the env otherwise steers the robot itself
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    if not isinstance(getattr(cfg.sim, "visualizer_cfgs", None), list):
        cfg.sim.visualizer_cfgs = []
    cfg.sim.visualizer_cfgs.append(KitVisualizerCfg(headless=True))

    os.makedirs(args_cli.out_dir, exist_ok=True)
    n_steps = int(args_cli.seconds / 0.02)
    cfg.video_recorders = [VideoRecorderCfg(
        source="visualizer:kit", output_dir=args_cli.out_dir, fps=args_cli.fps,
        video_length=n_steps,          # counts ENV STEPS, not captured frames
        video_interval=0, step_offset=0, frame_stride=args_cli.frame_stride)]

    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    robot = u.scene["robot"]
    sim = u.sim

    yaw0 = yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    W2S = np.array([[c0, -s0], [s0, c0]])
    print(f"spawn yaw {yaw0:+.3f}; rotating the street to match", flush=True)

    actor = build_actor(args_cli.policy, device)

    blob = json.loads(Path(args_cli.crowd).read_text())
    ts = blob["summary"]["suggested_time_scale"]
    start_xy = np.array(blob["summary"]["camera_path_m"])[0]
    goal_world = np.array(blob["summary"]["camera_path_m"])[-1]

    def to_sim(xy):
        return (W2S @ (np.asarray(xy) - start_xy).T).T

    goal_xy = to_sim(goal_world)

    tracks = []
    for tid, tr in blob["tracks"].items():
        tracks.append({"cls": tr["cls"], "r": tr["radius_m"],
                       "t": np.array(tr["t_s"]) * ts,
                       "P": to_sim(np.array(tr["xy_m"])),
                       "V": (W2S @ (np.array(tr["v_xy_mps_raw"]) / ts).T).T})
    all_y = np.concatenate([tr["P"][:, 1] for tr in tracks])
    corridor = (float(np.percentile(all_y, 5)), float(np.percentile(all_y, 95)))
    print(f"{len(tracks)} tracks; corridor y in [{corridor[0]:.2f},{corridor[1]:.2f}]",
          flush=True)

    # People and vehicles get different colours so the video is readable.
    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/World/crowd",
        markers={
            "person": sim_utils.CapsuleCfg(
                radius=0.25, height=1.15, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.42, 0.78))),
            "vehicle": sim_utils.CapsuleCfg(
                radius=0.45, height=1.0, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.27, 0.18))),
        }))

    cmd_term = u.command_manager._terms["base_velocity"]
    p = NavParams()
    dt = float(u.step_dt)
    clearances = []

    for step in range(n_steps):
        t = step * dt
        root = robot.data.root_pos_w[0].cpu().numpy()
        pos_xy = root[:2].copy()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
        vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()[:2]

        obstacles, trans, idx = [], [], []
        for tr in tracks:
            if t < tr["t"][0] or t > tr["t"][-1]:
                continue
            x = np.interp(t, tr["t"], tr["P"][:, 0])
            y = np.interp(t, tr["t"], tr["P"][:, 1])
            vx = np.interp(t, tr["t"], tr["V"][:, 0])
            vy = np.interp(t, tr["t"], tr["V"][:, 1])
            obstacles.append(Obstacle(np.array([x, y]), np.array([vx, vy]), tr["r"]))
            trans.append([x, y, 0.85 if tr["cls"] == "person" else 0.55])
            idx.append(0 if tr["cls"] == "person" else 1)

        if trans:
            markers.visualize(translations=torch.tensor(np.array(trans),
                                                        dtype=torch.float32, device=device),
                              marker_indices=torch.tensor(idx, device=device))

        cmd, info = command(pos_xy, yaw, vel_w, goal_xy, obstacles, p,
                            corridor=corridor)
        cmd_term.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32, device=device)

        with torch.no_grad():
            actions = actor(obs["policy"] if isinstance(obs, dict) else obs)
        obs, _, _, _, _ = env.step(actions)

        for ob in obstacles:
            clearances.append(float(np.linalg.norm(ob.pos - pos_xy)
                                    - ob.radius - p.robot_radius))

        if args_cli.cam == "chase":
            eye = (float(root[0] - 3.0 * np.cos(yaw)),
                   float(root[1] - 3.0 * np.sin(yaw)), float(root[2] + 1.6))
            tgt = (float(root[0] + 1.5 * np.cos(yaw)),
                   float(root[1] + 1.5 * np.sin(yaw)), float(root[2]))
        elif args_cli.cam == "side":
            eye = (float(root[0]), float(root[1]) - 7.0, 3.0)
            tgt = (float(root[0]), float(root[1]), 0.8)
        else:
            eye = (float(root[0]), float(root[1]), 16.0)
            tgt = (float(root[0]), float(root[1]), 0.0)
        sim.set_camera_view(eye=eye, target=tgt)

        if step % 250 == 0:
            print(f"  [{step}/{n_steps}] t={t:5.1f}s goal={np.linalg.norm(goal_xy-pos_xy):5.1f}m "
                  f"crowd={len(obstacles):2d} z={root[2]:.2f}", flush=True)

    c = np.array(clearances) if clearances else np.array([np.nan])
    print(f"VIDEO_DONE min_clearance={np.nanmin(c):.2f} collisions={int((c<0).sum())} "
          f"final_goal_dist={np.linalg.norm(goal_xy - robot.data.root_pos_w[0,:2].cpu().numpy()):.2f}")
    env.close()
    app.close()


main()
