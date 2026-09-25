"""M3: record the trained G1 walking on flat ground with scripted commands.

Command schedule (50Hz control):
    forward 5s -> left turn 3s -> strafe right 3s -> stop
Chase camera sits 2m behind and 1m above the robot base; the Kit viewport
camera is moved every step and the recorder captures that viewport.

Isaac Lab 24.2.4 notes (read off the installed package, not assumed):
  * ``render_mode="rgb_array"`` is deprecated and returns None. Frames are
    captured by ``VideoRecorderCfg`` on ``env_cfg.video_recorders``.
  * A concrete visualizer (KitVisualizerCfg) must be injected into
    ``env_cfg.sim.visualizer_cfgs`` BEFORE the simulation is launched.
  * ``RslRlVecEnvWrapper.get_observations()`` returns a TensorDict, not a
    tuple, and ``step()`` returns 4 values.
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out-dir", default="/root/densewalk/outputs/m3_raw")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--frame-stride", type=int, default=2,
                    help="capture 1 frame per N control steps; 50Hz control / stride 2\n"
                         "=> 300 frames covers the full 11s command schedule")
parser.add_argument("--add-ground", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

# a concrete Kit visualizer must exist before launch for recording to work
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402


def command_at(t: float) -> tuple[float, float, float]:
    """(lin_vel_x, lin_vel_y, ang_vel_z) for the scripted schedule."""
    if t < 5.0:
        return (1.0, 0.0, 0.0)      # forward
    if t < 8.0:
        return (0.5, 0.0, 0.8)      # left turn
    if t < 11.0:
        return (0.0, -0.5, 0.0)     # strafe right
    return (0.0, 0.0, 0.0)          # stop


def main() -> None:
    import importlib.metadata as _md

    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))

    if not isinstance(getattr(env_cfg.sim, "visualizer_cfgs", None), list):
        env_cfg.sim.visualizer_cfgs = []
    env_cfg.sim.visualizer_cfgs.append(KitVisualizerCfg(headless=True))

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=args_cli.out_dir,
            fps=args_cli.fps,
            # video_length counts ENV STEPS, not captured frames:
            # frames = video_length / frame_stride
            video_length=args_cli.frames * args_cli.frame_stride,
            video_interval=0,
            step_offset=0,
            frame_stride=args_cli.frame_stride,
        )
    ]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    if args_cli.add_ground:
        # visuals are never colliders; the G1 needs a real plane at z=0
        sim_utils.spawn_ground_plane("/World/ground", cfg=sim_utils.GroundPlaneCfg(visible=False))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[M3] loaded policy from {args_cli.checkpoint}", flush=True)

    sim = env.unwrapped.sim
    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    step_dt = env.unwrapped.step_dt
    print(f"[M3] control rate {1.0/step_dt:.1f} Hz", flush=True)

    obs = env.get_observations()
    heights: list[float] = []
    total_steps = args_cli.frames * args_cli.frame_stride + 40
    for step in range(total_steps):
        vx, vy, wz = command_at(step * step_dt)
        cmd_term.command[:, 0] = vx
        cmd_term.command[:, 1] = vy
        cmd_term.command[:, 2] = wz

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        q = robot.data.root_quat_w[0]
        root = robot.data.root_pos_w[0].cpu().numpy()
        heights.append(float(root[2]))
        yaw = float(torch.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)).cpu())
        eye = (float(root[0] - 2.0 * np.cos(yaw)),
               float(root[1] - 2.0 * np.sin(yaw)),
               float(root[2] + 1.0))
        sim.set_camera_view(eye=eye, target=(float(root[0]), float(root[1]), float(root[2])))

        if step % 50 == 0:
            print(f"[M3] step {step} z={root[2]:.3f} cmd=({vx},{vy},{wz})", flush=True)

    hz = np.array(heights)
    print(f"[M3] base height min={hz.min():.3f} mean={hz.mean():.3f} max={hz.max():.3f}")
    env.close()
    assert hz.min() > 0.2, f"G1 collapsed or fell through: min base z={hz.min():.3f}"
    print(f"[M3] recorder output in {args_cli.out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
