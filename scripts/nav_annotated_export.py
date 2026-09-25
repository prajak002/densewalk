"""Annotated navigation episodes for the viewer: what each robot perceives, the risk it faces,
and what it commands -- learned PPO policy vs rule-based TTC controller, identical episodes.

Per step and per perceived person: position, velocity, time-to-contact tau, predicted
closest-approach distance d_min and the repulsion weight w, exactly as
densewalk.crowd_nav.avoidance_vector computes them. For the learned policy also its
action a = pi(o) and its critic value V(o). The TTC terms are logged for both robots
(the rule-based one acts on them; for the learned one they describe the situation it faced).

Episode search: seeds where the learned robot reaches the goal, the rule-based robot
collides, the learned robot passes someone at 0.2-1.0 m and visibly deviates sideways
from the straight start->goal line -- i.e. avoidance you can see.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densewalk.crowd_env import CrowdEnv  # noqa: E402
from densewalk.crowd_nav import NavParams, avoidance_vector, command, time_to_collision, visible_obstacles  # noqa: E402


def person_terms(robot_pos, robot_vel, o, p: NavParams):
    rel_p, rel_v = o.pos - robot_pos, o.vel - robot_vel
    R = o.radius + p.robot_radius
    ttc = time_to_collision(rel_p, rel_v, R)
    vv = float(rel_v @ rel_v)
    t_star = float(np.clip(-(rel_p @ rel_v) / vv, 0.0, p.horizon_s)) if vv > 1e-9 else 0.0
    d_min = float(np.linalg.norm(rel_p + t_star * rel_v))
    crowd = max(0.0, (R + p.comfort_dist) - d_min)
    w = (1.0 / max(ttc, 0.2) if np.isfinite(ttc) else 0.0) + 0.6 * crowd
    return ttc, d_min, w, t_star


def run(env: CrowdEnv, seed, controller, model):
    obs = env.reset(seed=seed)
    p = env.p
    out = {"t0": env.t, "start": env.pos.tolist(), "goal": env.goal_xy.tolist(), "steps": []}
    done = trunc = False
    info = {}
    while not (done or trunc):
        crowd = env.crowd_at(env.t)
        seen, _ = visible_obstacles(env.pos, env.yaw, crowd, p)
        avoid, min_ttc = avoidance_vector(env.pos, env.vel, seen, p)
        rows = []
        for o in seen:
            ttc, dmin, w, _ = person_terms(env.pos, env.vel, o, p)
            rows.append([round(float(o.pos[0]), 2), round(float(o.pos[1]), 2), round(float(o.vel[0]), 2),
                         round(float(o.vel[1]), 2), round(float(o.radius), 2),
                         None if not np.isfinite(ttc) else round(ttc, 2), round(dmin, 2), round(w, 3)])
        value = None
        if controller == "learned":
            a, _ = model.predict(obs, deterministic=True)
            with torch.no_grad():
                value = float(model.policy.predict_values(model.policy.obs_to_tensor(obs)[0])[0, 0])
        else:
            cmd, _ = command(env.pos, env.yaw, env.vel, env.goal_xy, seen, p, corridor=env.corridor)
            a = np.array([cmd[0] / p.max_speed, cmd[1] / (p.max_speed * 0.5), cmd[2] / p.max_yaw_rate])
        pos, yaw = env.pos.copy(), env.yaw
        obs, _, done, trunc, info = env.step(a)
        a = np.clip(np.asarray(a, float), -1, 1)
        out["steps"].append({
            "t": round(env.t - env.dt, 3), "xy": np.round(pos, 3).tolist(), "yaw": round(float(yaw), 4),
            "cmd": [round(float(max(a[0], 0) * p.max_speed), 3), round(float(a[1] * p.max_speed * 0.5), 3),
                    round(float(a[2] * p.max_yaw_rate), 3)],
            "clr": None if info["clearance"] is None else round(info["clearance"], 3),
            "min_ttc": None if not np.isfinite(min_ttc) else round(float(min_ttc), 2),
            "avoid": np.round(avoid, 3).tolist(), "value": None if value is None else round(value, 2),
            "seen": rows})
    e = info
    out["outcome"] = "reached" if e.get("reached") else "collided" if e.get("collided") else "left street" if e.get("out_of_street") else "timeout"
    out["end_xy"] = np.round(env.pos, 3).tolist()
    clrs = [s["clr"] for s in out["steps"] if s["clr"] is not None]
    out["min_clear"] = round(min(clrs), 3) if clrs else None
    xy = np.array([s["xy"] for s in out["steps"]]); s0, g = np.array(out["start"]), np.array(out["goal"])
    d = (g - s0) / max(np.linalg.norm(g - s0), 1e-6)
    out["max_lateral_dev"] = round(float(np.abs((xy - s0) @ np.array([-d[1], d[0]])).max()), 2)
    out["close_steps"] = int(sum(1 for c in clrs if c < 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowd", required=True); ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=500); ap.add_argument("--seed0", type=int, default=6000)
    ap.add_argument("--seconds", type=float, default=40.0); ap.add_argument("--keep", type=int, default=4)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.policy, device="cpu")
    cands = []
    for k in range(a.seeds):
        s = a.seed0 + k
        L = run(CrowdEnv(a.crowd, seconds=a.seconds, seed=s), s, "learned", model)
        if L["outcome"] != "reached" or L["min_clear"] is None or not (0.2 <= L["min_clear"] <= 1.0):
            continue
        R = run(CrowdEnv(a.crowd, seconds=a.seconds, seed=s), s, "rule", model)
        score = L["max_lateral_dev"] + 0.1 * L["close_steps"] + (2.0 if R["outcome"] == "collided" else 0.0)
        cands.append((score, s, L, R))
    cands.sort(key=lambda c: -c[0])
    eps = [{"label": f"seed {s} · learned {L['outcome']}, rule {R['outcome']} · swerve {L['max_lateral_dev']} m, closest {L['min_clear']} m",
            "seed": s, "t0": L["t0"], "start": L["start"], "goal": L["goal"], "learned": L, "rule": R}
           for _, s, L, R in cands[: a.keep]]
    env = CrowdEnv(a.crowd, seconds=a.seconds, seed=0)
    crowd = [{"cls": tr["cls"], "r": tr["r"], "t": np.round(tr["t"], 3).tolist(), "xy": np.round(tr["P"], 3).tolist()}
             for tr in env.tracks]
    P = NavParams()
    Path(a.out).write_text(json.dumps({"params": {k: getattr(P, k) for k in P.__dataclass_fields__} | {"dt": env.dt},
                                       "corridor_y": list(env.corridor), "crowd": crowd, "episodes": eps},
                                      separators=(",", ":")))
    print(f"{len(cands)} candidate episodes (learned reached, passed someone at 0.2-1.0 m) of {a.seeds}")
    for e in eps:
        print(" ", e["label"])


if __name__ == "__main__":
    main()
