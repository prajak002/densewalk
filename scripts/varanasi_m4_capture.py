"""M4 capture pass: the G1 rendered alone, for compositing into the splat.

Isaac owns the robot and the physics; the splat owns the street. To see the
robot IN the street, render each separately from the same camera and combine.

The ground plane is made invisible (it still collides -- visuals are never
colliders, per the project invariants) and the camera background is set to a
key colour, so every non-key pixel is robot. Writes the RGB frames plus the
exact chase and eye poses per frame, so the splat pass can match them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--crowd", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--seconds", type=float, default=40.0)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
parser.add_argument("--cohort-shift", type=float, default=12.0,
                    help="metres between successive replays of the recorded crowd")
parser.add_argument("--cohorts", type=int, default=3,
                    help="how many replays overlap; more means a denser street")
parser.add_argument("--max-speed", type=float, default=1.0)
parser.add_argument("--comfort", type=float, default=0.6)
parser.add_argument("--stop-ttc", type=float, default=0.7)
parser.add_argument("--cam-back", type=float, default=3.2,
                    help="metres behind the robot")
parser.add_argument("--cam-up", type=float, default=1.5,
                    help="metres above the pelvis")
parser.add_argument("--cam-pitch", type=float, default=8.0,
                    help="degrees down. At 20deg the camera framed the ground "
                         "just ahead of the robot and cut off everyone beyond "
                         "~8m -- the very people it was avoiding.")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True}).app  # noqa: E402

import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk.crowd_nav import (NavParams, Obstacle, command,  # noqa: E402
                                 visible_obstacles)

KEY = (0.0, 1.0, 0.0)          # pure green: nothing in the G1 asset is this


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


def _cohort_obstacles(tracks, tq, shift_sim):
    """Obstacles from one replay of the recorded crowd, displaced along the street."""
    out = []
    for tr in tracks:
        if tq < tr["t"][0] or tq > tr["t"][-1]:
            continue
        out.append(Obstacle(
            np.array([np.interp(tq, tr["t"], tr["P"][:, 0]),
                      np.interp(tq, tr["t"], tr["P"][:, 1])]) + shift_sim,
            np.array([np.interp(tq, tr["t"], tr["V"][:, 0]),
                      np.interp(tq, tr["t"], tr["V"][:, 1])]), tr["r"]))
    return out


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    device = "cuda:0"
    cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    cfg.episode_length_s = args_cli.seconds + 10.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    # The camera must be part of the scene BEFORE the env is built. Creating a
    # standalone Camera afterwards and calling _initialize_impl() leaves it
    # outside the env's render loop: it initialises fine, updates without
    # error, and returns an empty buffer -- every frame came out 100% key
    # colour with no robot in it.
    # Mount the camera ON the robot's pelvis with a fixed offset, rather than
    # trying to drive a free camera each step. A scene sensor's pose comes from
    # cfg.offset relative to its parent prim, so set_world_poses_from_view was
    # silently ignored and the camera sat at the env origin -- INSIDE the G1's
    # leg, which is why every frame was a uniform colour at 0.11m depth.
    # Mounted, it follows the robot for free and its world pose is readable
    # from cam.data for the splat pass.
    cfg.scene.chase_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/chase_cam", update_period=0.0,
        width=args_cli.width, height=args_cli.height,
        data_types=["rgb", "distance_to_image_plane"], background_color=KEY,
        offset=CameraCfg.OffsetCfg(
            pos=(-args_cli.cam_back, 0.0, args_cli.cam_up),
            rot=(math.cos(math.radians(args_cli.cam_pitch) / 2), 0.0,
                 -math.sin(math.radians(args_cli.cam_pitch) / 2), 0.0),
            convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0,
                                         clipping_range=(0.05, 300.0)))

    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    robot = u.scene["robot"]

    # Hide the ground VISUALLY; it still collides. The splat supplies the street.
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    terrain_path = u.scene["terrain"].cfg.prim_path
    hidden = 0
    for prim in stage.Traverse():
        pstr = str(prim.GetPath())
        if pstr.startswith(terrain_path.replace(".*", "")) or "ground" in pstr.lower():
            img = UsdGeom.Imageable(prim)
            if img:
                img.MakeInvisible()
                hidden += 1
    print(f"hid {hidden} ground prims (collision unaffected)", flush=True)

    cam = u.scene["chase_cam"]

    actor = build_actor(args_cli.policy, device)

    blob = json.loads(Path(args_cli.crowd).read_text())
    ts = blob["summary"]["suggested_time_scale"]
    campath = np.array(blob["summary"]["camera_path_m"])
    start_xy, goal_world = campath[0], campath[-1]
    yaw0 = yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    W2S = np.array([[c0, -s0], [s0, c0]])

    def to_sim(xy):
        return (W2S @ (np.asarray(xy) - start_xy).T).T

    goal_xy = to_sim(goal_world)
    tracks = []
    for tid, tr in blob["tracks"].items():
        P_canon = np.array(tr["xy_m"])
        tracks.append({"cls": tr["cls"], "r": tr["radius_m"],
                       "t": np.array(tr["t_s"]) * ts,
                       "P_canon": P_canon,
                       "P": to_sim(P_canon),
                       "V": (W2S @ (np.array(tr["v_xy_mps_raw"]) / ts).T).T})
    # The clip gives ~17s of crowd (12s, speed-ramp corrected) but the robot
    crowd_period = max(float(tr["t"][-1]) for tr in tracks)
    # The clip recorded people walking TOWARD the camera, so once they pass the
    # robot the street ahead empties and there is nothing left to avoid --
    # clearance grew to 13m with nobody in view. Replay the recorded cohort
    # repeatedly, displaced along the street, so fresh oncoming traffic keeps
    # arriving. Unlike a modulo wrap this never drops anyone onto the robot.
    # One cohort at a time left the robot walking through gaps: with a 33.7m
    # shift it never reached the next group (it covers ~20m in 40s) and only
    # 85 of 1000 frames had anyone in view. Overlap consecutive cohorts and
    # space them by roughly how far a person walks in one period, so oncoming
    # traffic is continuous.
    # 18m spacing still left 90% of frames with nobody perceived: the robot
    # senses 12m and closes slowly, so it spent most of the walk between
    # groups. Space the replays inside sensing range and overlap five of them,
    # which is what a continuously busy street actually looks like.
    cohort_shift = args_cli.cohort_shift
    print(f"crowd period {crowd_period:.1f}s, cohort shift {cohort_shift:.1f}m",
          flush=True)

    all_y = np.concatenate([tr["P"][:, 1] for tr in tracks])
    corridor = (float(np.percentile(all_y, 5)), float(np.percentile(all_y, 95)))

    out = Path(args_cli.out_dir)
    (out / "robot").mkdir(parents=True, exist_ok=True)
    cmd_term = u.command_manager._terms["base_velocity"]
    p = NavParams(max_speed=args_cli.max_speed,
                  comfort_dist=args_cli.comfort,
                  stop_ttc=args_cli.stop_ttc)
    dt = float(u.step_dt)
    n_steps = int(args_cli.seconds / dt)
    meta = []
    import cv2

    saved = 0
    for step in range(n_steps):
        t = step * dt
        root = robot.data.root_pos_w[0].cpu().numpy()
        pos_xy = root[:2].copy()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
        vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()[:2]

        cohort = int(t // crowd_period)
        # Three overlapping replays keep people both ahead of and behind the
        # robot at all times, instead of one group that passes and leaves.
        obstacles = []
        half = args_cli.cohorts // 2
        for ck in range(cohort - half, cohort + half + 1):
            tqk = t - ck * crowd_period
            if tqk < 0:
                continue
            shift_sim = W2S @ np.array([ck * cohort_shift, 0.0])
            obstacles += _cohort_obstacles(tracks, tqk, shift_sim)

        # Navigate on what the robot can SEE, not on the full ground-truth
        # crowd: forward field of view, finite range, and people hidden behind
        # nearer people are simply not reported.
        seen, pinfo = visible_obstacles(pos_xy, yaw, obstacles, p)
        cmd, info = command(pos_xy, yaw, vel_w, goal_xy, seen, p,
                            corridor=corridor)
        cmd_term.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32, device=device)

        # chase camera: behind and above, looking slightly ahead of the robot
        eye = np.array([root[0] - 2.6 * np.cos(yaw), root[1] - 2.6 * np.sin(yaw),
                        root[2] + 1.25])
        tgt = np.array([root[0] + 1.2 * np.cos(yaw), root[1] + 1.2 * np.sin(yaw),
                        root[2] - 0.1])
        with torch.no_grad():
            actions = actor(obs["policy"] if isinstance(obs, dict) else obs)
        obs, _, _, _, _ = env.step(actions)

        if step % args_cli.stride == 0:
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            # With the ground hidden, anything at finite depth is the robot.
            # Colour keying proved unreliable here (the background did not come
            # back as the configured key colour), depth does not care.
            d = cam.data.output["distance_to_image_plane"][0].cpu().numpy()
            d = np.squeeze(d)
            mask = np.isfinite(d) & (d > 0.0) & (d < 60.0)
            rgba = np.dstack([rgb.astype(np.uint8),
                              (mask * 255).astype(np.uint8)])
            # This mount leaves the image with a 180 degree roll (the G1's
            # badge renders upside down). Composing the roll into the mount
            # quaternion moved the robot out of frame entirely, so correct it
            # in image space, where the result is exact and checkable.
            rgba = cv2.rotate(rgba, cv2.ROTATE_180)
            # Save depth too: a single assumed robot depth makes bodies
            # intersect, because half a person can be nearer than the robot's
            # torso while the rest is behind it. Millimetres in uint16.
            dmm = np.where(mask, np.clip(d * 1000.0, 0, 65000), 0).astype(np.uint16)
            cv2.imwrite(str(out / "robot" / f"{saved:05d}_depth.png"),
                        cv2.rotate(dmm, cv2.ROTATE_180))
            cv2.imwrite(str(out / "robot" / f"{saved:05d}.png"),
                        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            if saved == 0:
                print(f"  mask check: {mask.mean()*100:.2f}% robot pixels, "
                      f"depth range {np.nanmin(d[mask]) if mask.any() else -1:.2f}"
                      f"-{np.nanmax(d[mask]) if mask.any() else -1:.2f} m", flush=True)
            meta.append({
                "i": saved, "step": step, "t": t,
                "robot_xy_sim": [float(pos_xy[0]), float(pos_xy[1])],
                "robot_z": float(root[2]), "yaw_sim": yaw,
                "eye_sim": eye.tolist(), "target_sim": tgt.tolist(),
                "cam_pos_w": cam.data.pos_w[0].cpu().numpy().tolist(),
                "cam_quat_w": cam.data.quat_w_world[0].cpu().numpy().tolist(),
                "t_crowd": float(t - cohort * crowd_period),
                "cohort": int(cohort),
                "n_obstacles": len(obstacles),
                "n_visible": pinfo["n_visible"],
                "n_occluded": pinfo["n_occluded"],
                "n_out_of_view": pinfo["n_out_of_view"],
                "visible_xy": [[float(o.pos[0]), float(o.pos[1])] for o in seen],
                "all_xy": [[float(o.pos[0]), float(o.pos[1])] for o in obstacles],
                "goal_dist_m": float(info["goal_dist"]),
                "clearance_m": (None if not obstacles else
                                float(min(np.linalg.norm(o.pos - pos_xy)
                                          - o.radius - p.robot_radius
                                          for o in obstacles))),
                "stopped": bool(info["stopped"]),
            })
            saved += 1
        if step % 250 == 0:
            print(f"  [{step}/{n_steps}] saved={saved} goal={info['goal_dist']:.1f}m "
                  f"crowd={len(obstacles)} seen={pinfo['n_visible']} "
                  f"occluded={pinfo['n_occluded']}", flush=True)

    (out / "meta.json").write_text(json.dumps(
        {"start_xy": start_xy.tolist(), "spawn_yaw": float(yaw0),
         # the video pass must replay the crowd EXACTLY as the sim did, so the
         # cohort scheme travels with the capture rather than being guessed
         "cohort_shift": float(cohort_shift), "cohorts": int(args_cli.cohorts),
         # the video pass must use the SAME camera geometry or the composite
         # and the splat disagree about where the street is
         "cam_back": float(args_cli.cam_back), "cam_up": float(args_cli.cam_up),
         "cam_pitch_deg": float(args_cli.cam_pitch),
         "crowd_period": float(crowd_period),
         "key_color": KEY, "width": args_cli.width, "height": args_cli.height,
         "frames": meta}, indent=2))
    print(f"CAPTURE_DONE {saved} frames -> {out}")
    env.close()
    app.close()


main()
