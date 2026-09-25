"""Reactive navigation among moving people, for the G1 velocity policy.

The locomotion policy takes a body-frame velocity command
(v_x, v_y, omega_z) and handles the walking. This module decides WHAT to
command, given where the crowd is and where it is going.

Rule-based by design -- ORCA and learned planners are out of scope for this
project. The rule is time-to-collision, not distance: in a dense street almost
everyone is close, so a distance threshold either freezes the robot or ignores
everyone. What matters is who is closing, and how soon.

Everything here is plain numpy in the canonical frame (metres, Z-up, floor
z=0) so it can be tested without launching a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """A moving obstacle on the ground plane."""
    pos: np.ndarray      # (2,) metres
    vel: np.ndarray      # (2,) m/s
    radius: float        # metres


@dataclass(frozen=True)
class NavParams:
    max_speed: float = 1.0          # the G1 policy was trained around this
    max_yaw_rate: float = 0.8       # rad/s
    robot_radius: float = 0.35      # G1 footprint plus margin
    horizon_s: float = 3.0          # how far ahead a collision still matters
    goal_gain: float = 1.0
    # Avoidance must be able to override goal-seeking, but not swamp it. At
    # gain 2.2 against a capped sum of 1.6 the repulsion reached 3.5x the goal
    # term, so in oncoming traffic it saturated, pointed backwards, got clamped
    # to pure lateral, and the robot sidestepped for 35s without advancing
    # (0.16 m/s of an allowed 1.0). Comparable magnitudes let it thread rather
    # than flee.
    avoid_gain: float = 1.2
    comfort_dist: float = 0.6       # personal space beyond the radii
    stop_ttc: float = 0.7           # below this, stop rather than swerve
    max_avoid: float = 1.0          # cap on the summed repulsion, see below
    wall_gain: float = 2.5          # how hard the street edges push back
    wall_margin: float = 0.5        # start pushing this far inside the edge
    max_deflect_rad: float = 1.31   # 75 deg: how far avoidance may bend the
                                    # heading away from the goal direction
    fov_rad: float = 2.09           # 120 deg horizontal: what the robot can see
    sense_range_m: float = 12.0     # beyond this a 640x360 render gives nothing
                                    # a detector could act on


def time_to_collision(rel_pos: np.ndarray, rel_vel: np.ndarray,
                      combined_radius: float) -> float:
    """Time until two discs of combined radius touch, or inf if they never do.

    Solves |rel_pos + t*rel_vel| = R for the smallest positive t. Returns 0.0
    when already overlapping, which the caller must treat as an emergency
    rather than as 'plenty of time'.
    """
    a = float(rel_vel @ rel_vel)
    b = 2.0 * float(rel_pos @ rel_vel)
    c = float(rel_pos @ rel_pos) - combined_radius ** 2
    if c <= 0.0:
        return 0.0
    if a < 1e-9:
        return np.inf          # no relative motion: never closes
    disc = b * b - 4 * a * c
    if disc <= 0.0:
        return np.inf
    root = (-b - np.sqrt(disc)) / (2 * a)
    return float(root) if root > 0 else np.inf


def avoidance_vector(robot_pos: np.ndarray, robot_vel: np.ndarray,
                     obstacles: list[Obstacle], p: NavParams):
    """Sum of repulsions from obstacles we are actually closing with.

    Returns (vector, min_ttc). The repulsion is scaled by 1/ttc so someone
    about to step in front dominates someone merely nearby, and is directed
    away from the predicted contact point rather than from the current
    position -- otherwise the robot dodges where a person WAS.
    """
    total = np.zeros(2)
    min_ttc = np.inf
    for ob in obstacles:
        rel_p = ob.pos - robot_pos
        rel_v = ob.vel - robot_vel
        # TTC uses the HARD radius (bodies actually touching). Folding the
        # comfort margin in here instead makes every pedestrian within ~1.2m
        # read as an active collision, and in a street this dense that is
        # always true -- the robot then emergency-stops forever and never
        # moves. Comfort shapes the repulsion below; it is not an emergency.
        R_hard = ob.radius + p.robot_radius
        ttc = time_to_collision(rel_p, rel_v, R_hard)
        # Crowding is judged at the PREDICTED closest approach, not at the
        # current distance. Someone 3.3m away who will cut to within 1m of us
        # in two seconds is the situation worth reacting to; judging by present
        # distance ignores them until it is too late to do anything gentle.
        t_star = 0.0
        vv = float(rel_v @ rel_v)
        if vv > 1e-9:
            t_star = float(np.clip(-(rel_p @ rel_v) / vv, 0.0, p.horizon_s))
        d_min = float(np.linalg.norm(rel_p + t_star * rel_v))
        crowding = max(0.0, (R_hard + p.comfort_dist) - d_min)
        if ttc > p.horizon_s and crowding <= 0.0:
            continue
        min_ttc = min(min_ttc, ttc)
        # where they will be when it matters
        t_eval = min(ttc, p.horizon_s) if np.isfinite(ttc) else t_star
        future = (ob.pos + ob.vel * t_eval) - (robot_pos + robot_vel * t_eval)
        dist = float(np.linalg.norm(future))
        if dist < 1e-6:
            # exactly head-on: break the tie sideways, deterministically
            away = np.array([-rel_p[1], rel_p[0]])
            away = away / max(float(np.linalg.norm(away)), 1e-6)
        else:
            away = -future / dist
        # closing soon dominates; merely being crowded adds a gentle push
        weight = (1.0 / max(ttc, 0.2)) if np.isfinite(ttc) else 0.0
        # Crowding is a nudge, not an alarm: in a market someone is always
        # inside the comfort radius, and at 1.5 this term alone saturated the
        # cap and left no headroom for the ttc term that actually matters.
        weight += 0.6 * crowding
        total += away * weight
    # Cap the summed repulsion. Seventeen pedestrians each pushing gently sum
    # to a shove that overwhelms goal-seeking entirely -- in the first run the
    # robot was driven sideways out of the street and never came back. Capping
    # keeps avoidance authoritative without letting crowd density alone decide
    # where the robot goes.
    mag = float(np.linalg.norm(total))
    if mag > p.max_avoid:
        total = total / mag * p.max_avoid
    return total, min_ttc


def visible_obstacles(robot_pos: np.ndarray, robot_yaw: float,
                      obstacles: list[Obstacle],
                      p: NavParams = NavParams()) -> tuple[list[Obstacle], dict]:
    """The subset of the crowd the robot can actually perceive.

    Handing the planner every obstacle in the scene is not navigation among
    people, it is navigation with omniscience -- the robot dodges someone
    walking up behind it, which no camera could support. Three limits, all of
    them properties of a forward-facing camera:

      field of view   nothing outside +/- fov/2 of the heading
      range           nothing beyond sense_range_m
      occlusion       nothing hidden behind a nearer person

    Occlusion is the one that matters in a dense street: the robot routinely
    cannot see the person behind the person in front, which is precisely the
    regime this project set out to study.
    """
    cand = []
    for ob in obstacles:
        rel = ob.pos - robot_pos
        dist = float(np.linalg.norm(rel))
        if dist > p.sense_range_m:
            continue
        ang = np.arctan2(rel[1], rel[0]) - robot_yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        if abs(ang) > p.fov_rad / 2:
            continue
        half = float(np.arcsin(np.clip(ob.radius / max(dist, 1e-6), -1.0, 1.0)))
        cand.append((dist, ang, half, ob))

    cand.sort(key=lambda z: z[0])       # nearest first: only they can occlude
    seen, blockers = [], []
    n_occluded = 0
    for dist, ang, half, ob in cand:
        hidden = any(abs(ang - a2) + half < h2 for a2, h2 in blockers)
        if hidden:
            n_occluded += 1
            continue
        seen.append(ob)
        blockers.append((ang, half))
    return seen, {"n_total": len(obstacles), "n_visible": len(seen),
                  "n_occluded": n_occluded,
                  "n_out_of_view": len(obstacles) - len(cand)}


def wall_push(robot_pos: np.ndarray, bounds: tuple[float, float] | None,
              p: NavParams) -> np.ndarray:
    """Keep the robot inside the street.

    The buildings exist in the splat but the splat is visual only, so without
    this the robot is free to walk out through a shopfront -- which is exactly
    what it did before this existed. The corridor comes from the same
    reconstruction, so this is the real street's width, not a made-up box.
    """
    if bounds is None:
        return np.zeros(2)
    y_lo, y_hi = bounds
    y = float(robot_pos[1])
    push = 0.0
    if y > y_hi - p.wall_margin:
        push = -(y - (y_hi - p.wall_margin))
    elif y < y_lo + p.wall_margin:
        push = ((y_lo + p.wall_margin) - y)
    return np.array([0.0, push * p.wall_gain])


def command(robot_pos: np.ndarray, robot_yaw: float, robot_vel: np.ndarray,
            goal: np.ndarray, obstacles: list[Obstacle],
            p: NavParams = NavParams(),
            corridor: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """Body-frame (v_x, v_y, omega_z) for the locomotion policy.

    The returned dict carries the reasoning (min ttc, whether it stopped)
    so a run can be inspected afterwards instead of guessed at.
    """
    to_goal = goal - robot_pos
    goal_dist = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / goal_dist if goal_dist > 1e-6 else np.zeros(2)

    avoid, min_ttc = avoidance_vector(robot_pos, robot_vel, obstacles, p)

    walls = wall_push(robot_pos, corridor, p)
    desired = p.goal_gain * goal_dir + p.avoid_gain * avoid + walls

    # Keep the heading anchored to the goal. Steering along the FULL desired
    # vector lets a backwards-pointing repulsion spin the robot 180 degrees;
    # the no-reversing clamp is in body frame, so it then walks "forward" away
    # from the goal and oscillates without ever advancing. Allowing avoidance
    # to deflect the heading by at most max_deflect_rad means it can step
    # around people -- even sharply -- but never turn its back on the goal.
    if float(np.linalg.norm(desired)) > 1e-6 and goal_dist > 1e-6:
        ang_goal = np.arctan2(goal_dir[1], goal_dir[0])
        ang_des = np.arctan2(desired[1], desired[0])
        diff = (ang_des - ang_goal + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) > p.max_deflect_rad:
            diff = np.sign(diff) * p.max_deflect_rad
            mag = float(np.linalg.norm(desired))
            ang = ang_goal + diff
            desired = mag * np.array([np.cos(ang), np.sin(ang)])

    speed = float(np.linalg.norm(desired))

    stopped = False
    if min_ttc < p.stop_ttc:
        # Too late to steer out of it. A humanoid that stops is recoverable;
        # one that swerves into someone else is not.
        desired = np.zeros(2)
        speed = 0.0
        stopped = True
    elif speed > 1e-6:
        desired = desired / speed * min(p.max_speed, speed * p.max_speed)

    # world -> body frame
    c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
    R = np.array([[c, -s], [s, c]])
    v_body = R @ desired

    # Never reverse -- but do not freeze either. Zeroing the whole command
    # whenever the repulsion pointed backwards made the robot stand still for
    # 95% of a 35s run in oncoming traffic: in a head-on crowd the sum points
    # backwards most of the time. Clamp only the backward component and KEEP
    # the lateral one, which is how a person actually gets through a crowd --
    # sidestepping into the gaps rather than stopping dead. A true stop is
    # reserved for imminent contact (stop_ttc above).
    if v_body[0] < 0.0:
        v_body = np.array([0.0, v_body[1]])

    # face where we are going; the policy walks better forward than sideways
    if not stopped and np.linalg.norm(desired) > 1e-6:
        heading_err = np.arctan2(desired[1], desired[0]) - robot_yaw
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        omega = float(np.clip(2.0 * heading_err, -p.max_yaw_rate, p.max_yaw_rate))
    else:
        omega = 0.0

    return (np.array([v_body[0], v_body[1], omega]),
            {"min_ttc": float(min_ttc), "stopped": stopped,
             "goal_dist": goal_dist, "wall_push": float(np.linalg.norm(walls))})
