"""Learned (PPO) vs rule-based (TTC) navigation on identical Varanasi crowd episodes, for the viewer.

Both controllers run in densewalk.crowd_env.CrowdEnv -- the exact environment the
PPO policy (outputs/varanasi/nav_policy/nav_policy.zip) was trained in -- with the
same seed per episode, so start, goal, start time and crowd are identical and the
only difference is the controller. The robot is modelled at the velocity-command
level (what the controller outputs and the G1 locomotion policy consumes);
locomotion itself is not simulated here.

Writes:
  nav_compare.json   summary over all episodes + full logs of the showcase episodes
                     (per step: t, pos, yaw, command, clearance, ids of the people
                     the robot can perceive), and the env's cleaned crowd paths
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk.crowd_env import CrowdEnv  # noqa: E402
from densewalk.crowd_nav import NavParams, command, visible_obstacles  # noqa: E402


def run(env: CrowdEnv, seed: int, controller, model=None):
    obs = env.reset(seed=seed)
    p = env.p
    log = {"t0": env.t, "start": env.pos.tolist(), "goal": env.goal_xy.tolist(), "steps": []}
    done = trunc = False
    info = {}
    while not (done or trunc):
        crowd = env.crowd_at(env.t)
        seen, _ = visible_obstacles(env.pos, env.yaw, crowd, p)
        seen_ids = [i for i, o in enumerate(crowd) if any(o is s for s in seen)]
        if controller == "learned":
            a, _ = model.predict(obs, deterministic=True)
        else:
            cmd, _ = command(env.pos, env.yaw, env.vel, env.goal_xy, seen, p, corridor=env.corridor)
            a = np.array([cmd[0] / p.max_speed, cmd[1] / (p.max_speed * 0.5), cmd[2] / p.max_yaw_rate])
        pos, yaw = env.pos.copy(), env.yaw
        obs, _, done, trunc, info = env.step(a)
        a = np.clip(np.asarray(a, float), -1, 1)
        log["steps"].append({
            "t": round(env.t - env.dt, 3), "xy": np.round(pos, 3).tolist(), "yaw": round(float(yaw), 4),
            "cmd": [round(float(max(a[0], 0) * p.max_speed), 3), round(float(a[1] * p.max_speed * 0.5), 3),
                    round(float(a[2] * p.max_yaw_rate), 3)],
            "clr": None if info["clearance"] is None else round(info["clearance"], 3),
            "seen": [np.round(crowd[i].pos, 2).tolist() for i in seen_ids],
        })
    log["end"] = {"xy": np.round(env.pos, 3).tolist(), "reached": info.get("reached", False),
                  "collided": info.get("collided", False), "out": info.get("out_of_street", False),
                  "goal_dist": round(info.get("goal_dist", float("nan")), 3)}
    clrs = [s["clr"] for s in log["steps"] if s["clr"] is not None]
    log["min_clear"] = round(min(clrs), 3) if clrs else None
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowd", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=5000)
    args = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(args.policy, device="cpu")

    env_l = CrowdEnv(args.crowd, seconds=args.seconds, seed=args.seed)
    env_r = CrowdEnv(args.crowd, seconds=args.seconds, seed=args.seed)
    pairs = []
    for ep in range(args.episodes):
        s = args.seed + ep
        L, R = run(env_l, s, "learned", model), run(env_r, s, "rule")
        assert np.allclose(L["start"], R["start"]) and abs(L["t0"] - R["t0"]) < 1e-9, "episodes differ"
        pairs.append((s, L, R))

    def summ(k):
        logs = [p[1 if k == "learned" else 2] for p in pairs]
        n = len(logs)
        mc = [l["min_clear"] for l in logs if l["min_clear"] is not None]
        return {"episodes": n,
                "success_rate": round(sum(l["end"]["reached"] for l in logs) / n, 3),
                "collision_rate": round(sum(l["end"]["collided"] for l in logs) / n, 3),
                "left_street_rate": round(sum(l["end"]["out"] for l in logs) / n, 3),
                "mean_min_clearance_m": round(float(np.mean(mc)), 3) if mc else None,
                "mean_steps": round(float(np.mean([len(l["steps"]) for l in logs])), 1)}

    def outcome(l):
        e = l["end"]
        return "reached" if e["reached"] else "collided" if e["collided"] else "left street" if e["out"] else "timeout"

    # showcases: the controllers disagree, both ways, plus one where both succeed
    cats = {"learned wins": [], "rule wins": [], "both reach": []}
    for s, L, R in pairs:
        ol, orr = outcome(L), outcome(R)
        if ol == "reached" and orr == "collided":
            cats["learned wins"].append((s, L, R))
        elif orr == "reached" and ol == "collided":
            cats["rule wins"].append((s, L, R))
        elif ol == orr == "reached":
            cats["both reach"].append((s, L, R))
    show = []
    for name, lst in cats.items():
        # prefer longer episodes: more crowd interaction to watch
        for s, L, R in sorted(lst, key=lambda x: -(len(x[1]["steps"]) + len(x[2]["steps"])))[:2]:
            show.append({"label": f"{name} (seed {s})", "seed": s, "t0": L["t0"], "start": L["start"], "goal": L["goal"],
                         "learned": L | {"outcome": outcome(L)}, "rule": R | {"outcome": outcome(R)}})

    crowd = [{"cls": tr["cls"], "r": tr["r"], "t": np.round(tr["t"], 3).tolist(), "xy": np.round(tr["P"], 3).tolist()}
             for tr in env_l.tracks]
    out = {"what": "CrowdEnv rollouts, identical seeds; robot at velocity-command level (locomotion not simulated)",
           "summary": {"learned (PPO)": summ("learned"), "rule-based (TTC)": summ("rule"),
                       "disagreements": {k: len(v) for k, v in cats.items()}},
           "params": {"fov_rad": NavParams().fov_rad, "sense_range_m": NavParams().sense_range_m,
                      "robot_radius": NavParams().robot_radius, "dt": env_l.dt},
           "corridor_y": list(env_l.corridor), "crowd": crowd, "episodes": show}
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps(out["summary"], indent=2))
    print("showcases:", [e["label"] for e in show])


if __name__ == "__main__":
    main()
