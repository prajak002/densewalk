"""Training environment: learn to walk forward through the real Varanasi crowd.

The crowd is not synthetic and not scripted -- every pedestrian, bicycle and
motorbike here is a track lifted from the source video into metric ground
coordinates. The robot learns against the people who were actually in that
street.

What the agent controls is the VELOCITY COMMAND, not the joints: the G1's
locomotion policy (outputs/g1_policy, 1500 iters) already walks and is reused
unchanged. So this env models the robot kinematically at the command level,
which is both what the locomotion policy consumes and fast enough to train on.

What the agent observes is only what it could perceive -- forward field of
view, limited range, and people hidden behind nearer people are not reported.
Training on the full ground-truth crowd would teach it to dodge pedestrians
walking up behind it, a skill no camera can support.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from densewalk.crowd_nav import NavParams, Obstacle, visible_obstacles

MAX_HUMAN_SPEED = 2.5     # m/s; anything faster is tracking noise, not walking


def _clean_path(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove back-projection jitter from a tracked trajectory.

    Foot points back-projected at distance are noisy: some tracks jump 5.5m in
    a 0.1s step (55 m/s), which is the far-field depth error, not a person. Left
    in, those jumps teleport a pedestrian onto the robot and register as a
    collision nothing could have avoided. Clamp each step to a human speed, then
    smooth lightly.
    """
    def clamp(X):
        X = X.copy()
        for i in range(1, len(X)):
            dt = max(float(t[i] - t[i - 1]), 1e-6)
            d = X[i] - X[i - 1]
            n = float(np.linalg.norm(d))
            lim = MAX_HUMAN_SPEED * dt
            if n > lim:
                X[i] = X[i - 1] + d / n * lim
        return X

    Q = clamp(P.astype(float))
    if len(Q) >= 5:
        # Edge-replicate before smoothing. np.convolve(mode="same") zero-pads,
        # which drags the first and last samples toward the origin and creates
        # a larger jump than the one being removed.
        k = np.ones(5) / 5.0
        for a in range(2):
            padded = np.pad(Q[:, a], 2, mode="edge")
            Q[:, a] = np.convolve(padded, k, mode="valid")
        # Smoothing across unevenly spaced samples can reintroduce a step, so
        # clamp once more after it.
        Q = clamp(Q)
    return Q


N_SLOTS = 6          # nearest visible neighbours fed to the policy
OBS_PER_SLOT = 5     # dx, dy, dvx, dvy, radius (body frame)


class CrowdEnv:
    """Gymnasium-style env; kept dependency-light so it runs in any of the venvs."""

    def __init__(self, crowd_json: str | Path, seconds: float = 40.0,
                 dt: float = 0.1, params: NavParams | None = None,
                 seed: int = 0, randomize_start: bool = True):
        blob = json.loads(Path(crowd_json).read_text())
        self.summary = blob["summary"]
        ts = self.summary["suggested_time_scale"]
        self.tracks = []
        for tr in blob["tracks"].values():
            t = np.array(tr["t_s"]) * ts
            P = _clean_path(np.array(tr["xy_m"]), t)
            V = np.gradient(P, axis=0) / np.maximum(np.gradient(t), 1e-6)[:, None]
            self.tracks.append({"cls": tr["cls"], "r": float(tr["radius_m"]),
                                "t": t, "P": P, "V": V})
        self.period = max(float(t["t"][-1]) for t in self.tracks)
        self.cam_path = np.array(self.summary["camera_path_m"])
        self.start_xy, self.goal_xy = self.cam_path[0], self.cam_path[-1]
        ys = np.concatenate([t["P"][:, 1] for t in self.tracks])
        self.corridor = (float(np.percentile(ys, 5)), float(np.percentile(ys, 95)))

        self.p = params or NavParams()
        self.dt = dt
        self.max_steps = int(seconds / dt)
        self.rng = np.random.default_rng(seed)
        self.randomize_start = randomize_start
        # 3 goal + 2 body velocity + 2 kerb distances, then the slots
        self.obs_dim = 7 + N_SLOTS * OBS_PER_SLOT
        self.act_dim = 3

    # ------------------------------------------------------------------ crowd
    def crowd_at(self, t: float) -> list[Obstacle]:
        tq = t          # no wrap: see reset() on the teleport artefact
        out = []
        for tr in self.tracks:
            if tq < tr["t"][0] or tq > tr["t"][-1]:
                continue
            out.append(Obstacle(
                np.array([np.interp(tq, tr["t"], tr["P"][:, 0]),
                          np.interp(tq, tr["t"], tr["P"][:, 1])]),
                np.array([np.interp(tq, tr["t"], tr["V"][:, 0]),
                          np.interp(tq, tr["t"], tr["V"][:, 1])]),
                tr["r"]))
        return out

    # ------------------------------------------------------------------- core
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_i = 0
        # Do NOT loop the crowd inside an episode. The naive modulo wrap
        # teleports every pedestrian back to their start position at the period
        # boundary, and someone rematerialises on top of the robot: clearance
        # went 14.96m -> -0.59m in a single step and both controllers scored a
        # 100% collision rate against a discontinuity rather than a person.
        # Instead keep episodes shorter than the crowd period and vary WHERE
        # along the street the robot starts, which gives more varied situations
        # anyway.
        span = self.max_steps * self.dt
        for _ in range(64):
            self.t = (float(self.rng.uniform(0, max(self.period - span, 1e-3)))
                      if self.randomize_start else 0.0)
            if self.randomize_start:
                k = int(self.rng.integers(0, max(len(self.cam_path) - 40, 1)))
                self.pos = self.cam_path[k].astype(float) + self.rng.normal(0, 0.4, 2)
                # Episodes are capped below the crowd period (~17s) and the
                # robot tops out near 1 m/s, so goals further than ~14m are not
                # reachable and would score as failures regardless of skill.
                far = min(k + int(self.rng.integers(80, 165)), len(self.cam_path) - 1)
                self.goal_xy = self.cam_path[far].astype(float)
            else:
                self.pos = self.cam_path[0].astype(float).copy()
                self.goal_xy = self.cam_path[-1].astype(float).copy()
            crowd = self.crowd_at(self.t)
            clear = min((float(np.linalg.norm(o.pos - self.pos))
                         - o.radius - self.p.robot_radius) for o in crowd) \
                if crowd else np.inf
            if clear > self.p.comfort_dist:
                break

        d = self.goal_xy - self.pos
        self.yaw = float(np.arctan2(d[1], d[0]))
        self.vel = np.zeros(2)
        self.prev_goal_dist = float(np.linalg.norm(self.goal_xy - self.pos))
        return self._obs()

    def _obs(self) -> np.ndarray:
        crowd = self.crowd_at(self.t)
        seen, _ = visible_obstacles(self.pos, self.yaw, crowd, self.p)
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        R = np.array([[c, -s], [s, c]])          # world -> body

        to_goal = R @ (self.goal_xy - self.pos)
        gd = float(np.linalg.norm(to_goal))
        head = [to_goal[0] / max(gd, 1e-6), to_goal[1] / max(gd, 1e-6),
                min(gd / 30.0, 2.0)]
        vb = R @ self.vel
        # distance to each kerb, so the policy can learn to stay in the street
        walls = [(self.corridor[1] - self.pos[1]) / 5.0,
                 (self.pos[1] - self.corridor[0]) / 5.0]
        vec = head + [vb[0], vb[1]] + walls

        seen.sort(key=lambda o: np.linalg.norm(o.pos - self.pos))
        for i in range(N_SLOTS):
            if i < len(seen):
                o = seen[i]
                dp = R @ (o.pos - self.pos)
                dv = R @ (o.vel - self.vel)
                vec += [dp[0] / 12.0, dp[1] / 12.0, dv[0] / 3.0, dv[1] / 3.0,
                        o.radius]
            else:
                # a slot with no one in it must be unmistakably "empty", not
                # "someone at the origin" -- park it far ahead with zero size
                vec += [1.5, 0.0, 0.0, 0.0, 0.0]
        return np.asarray(vec, dtype=np.float32)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        v_body = np.array([a[0] * self.p.max_speed, a[1] * self.p.max_speed * 0.5])
        omega = float(a[2] * self.p.max_yaw_rate)

        # the locomotion policy walks forward far better than it walks backwards
        v_body[0] = max(v_body[0], 0.0)

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rb = np.array([[c, -s], [s, c]])         # body -> world
        self.vel = Rb @ v_body
        self.pos = self.pos + self.vel * self.dt
        self.yaw = float((self.yaw + omega * self.dt + np.pi) % (2 * np.pi) - np.pi)
        self.t += self.dt
        self.step_i += 1

        crowd = self.crowd_at(self.t)
        clear = np.inf
        for o in crowd:
            clear = min(clear, float(np.linalg.norm(o.pos - self.pos)
                                     - o.radius - self.p.robot_radius))

        gd = float(np.linalg.norm(self.goal_xy - self.pos))
        progress = self.prev_goal_dist - gd
        self.prev_goal_dist = gd

        # Reward: make progress, keep people at arm's length, stay in the
        # street, and do not dither. The collision term is large and terminal
        # because a humanoid that walks into someone has failed outright, not
        # slightly.
        r = 6.0 * progress - 0.01
        if np.isfinite(clear) and clear < self.p.comfort_dist:
            r -= 1.2 * (self.p.comfort_dist - clear)
        y = self.pos[1]
        if y > self.corridor[1] or y < self.corridor[0]:
            r -= 0.6
        r -= 0.02 * abs(omega)

        collided = bool(np.isfinite(clear) and clear < 0.0)
        reached = gd < 1.0
        out_of_street = bool(y > self.corridor[1] + 2.5 or y < self.corridor[0] - 2.5)
        if collided:
            r -= 12.0
        if reached:
            r += 25.0

        done = collided or reached or out_of_street
        truncated = self.step_i >= self.max_steps
        info = {"collided": collided, "reached": reached,
                "out_of_street": out_of_street, "goal_dist": gd,
                "clearance": float(clear) if np.isfinite(clear) else None,
                "n_crowd": len(crowd)}
        return self._obs(), float(r), done, truncated, info
