"""Train the G1 to walk forward through the real crowd without colliding.

The locomotion policy is NOT retrained -- it already walks (1500 iters, mean
reward 28.09) and the project treats it as fixed. What is learned here is the
navigation policy: given what the robot can perceive of the crowd, what
velocity command to send. That is the part that was hand-written rules, and
that is the part the crowd data can actually teach.

Trained against the recorded Varanasi crowd, under the robot's real perception
limits (forward FOV, finite range, people occluded behind nearer people), and
evaluated head to head against the rule-based controller on identical episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym
from gymnasium import spaces

from densewalk.crowd_env import CrowdEnv
from densewalk.crowd_nav import NavParams, command, visible_obstacles


class CrowdGym(gym.Env):
    """Thin gymnasium shell around CrowdEnv."""

    metadata = {"render_modes": []}

    def __init__(self, crowd_json: str, seconds: float = 40.0, seed: int = 0):
        super().__init__()
        self.core = CrowdEnv(crowd_json, seconds=seconds, seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (self.core.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.core.act_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.core.reset(seed), {}

    def step(self, action):
        obs, r, done, trunc, info = self.core.step(action)
        return obs, r, done, trunc, info


def rollout_rule_based(crowd_json: str, episodes: int, seconds: float, seed: int):
    """Baseline: the hand-written time-to-collision controller."""
    env = CrowdEnv(crowd_json, seconds=seconds, seed=seed)
    p = NavParams()
    stats = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = trunc = False
        info = {}
        min_clear, steps = np.inf, 0
        while not (done or trunc):
            crowd = env.crowd_at(env.t)
            seen, _ = visible_obstacles(env.pos, env.yaw, crowd, p)
            cmd, _ = command(env.pos, env.yaw, env.vel, env.goal_xy, seen, p,
                             corridor=env.corridor)
            a = np.array([cmd[0] / p.max_speed,
                          cmd[1] / (p.max_speed * 0.5),
                          cmd[2] / p.max_yaw_rate])
            _, _, done, trunc, info = env.step(a)
            if info["clearance"] is not None:
                min_clear = min(min_clear, info["clearance"])
            steps += 1
        stats.append({"reached": info.get("reached", False),
                      "collided": info.get("collided", False),
                      "out": info.get("out_of_street", False),
                      "goal_dist": info.get("goal_dist"),
                      "min_clear": None if not np.isfinite(min_clear) else min_clear,
                      "steps": steps})
    return stats


def rollout_policy(model, crowd_json: str, episodes: int, seconds: float, seed: int):
    env = CrowdEnv(crowd_json, seconds=seconds, seed=seed)
    stats = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = trunc = False
        info = {}
        min_clear, steps = np.inf, 0
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(a)
            if info["clearance"] is not None:
                min_clear = min(min_clear, info["clearance"])
            steps += 1
        stats.append({"reached": info.get("reached", False),
                      "collided": info.get("collided", False),
                      "out": info.get("out_of_street", False),
                      "goal_dist": info.get("goal_dist"),
                      "min_clear": None if not np.isfinite(min_clear) else min_clear,
                      "steps": steps})
    return stats


def summarise(name, stats):
    n = len(stats)
    clears = [s["min_clear"] for s in stats if s["min_clear"] is not None]
    return {
        "controller": name, "episodes": n,
        "success_rate": round(sum(s["reached"] for s in stats) / n, 3),
        "collision_rate": round(sum(s["collided"] for s in stats) / n, 3),
        "left_street_rate": round(sum(s["out"] for s in stats) / n, 3),
        "mean_final_goal_dist_m": round(float(np.mean([s["goal_dist"] for s in stats])), 2),
        "mean_min_clearance_m": round(float(np.mean(clears)), 3) if clears else None,
        "mean_steps": round(float(np.mean([s["steps"] for s in stats])), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--eval-episodes", type=int, default=60)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def mk(i):
        return lambda: CrowdGym(args.crowd, seconds=args.seconds, seed=1000 + i)

    venv = VecMonitor(DummyVecEnv([mk(i) for i in range(args.n_envs)]))
    model = PPO("MlpPolicy", venv, verbose=1, n_steps=512, batch_size=1024,
                gae_lambda=0.95, gamma=0.995, learning_rate=3e-4,
                ent_coef=0.004, clip_range=0.2,
                policy_kwargs=dict(net_arch=[128, 128]))
    print(f"training PPO for {args.steps} steps on {args.n_envs} envs", flush=True)
    model.learn(total_timesteps=args.steps, progress_bar=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved {out}.zip", flush=True)

    print("evaluating head to head on identical episodes ...", flush=True)
    learned = summarise("learned (PPO)",
                        rollout_policy(model, args.crowd, args.eval_episodes,
                                       args.seconds, seed=7000))
    rule = summarise("rule-based (TTC)",
                     rollout_rule_based(args.crowd, args.eval_episodes,
                                        args.seconds, seed=7000))
    report = {"learned": learned, "rule_based": rule}
    Path(str(out) + "_eval.json").write_text(json.dumps(report, indent=2))
    print("EVAL " + json.dumps(report))


if __name__ == "__main__":
    main()
