"""Tests for the crowd training environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from densewalk.crowd_env import N_SLOTS, OBS_PER_SLOT, CrowdEnv

CROWD = Path("outputs/varanasi/crowd_world.json")
pytestmark = pytest.mark.skipif(not CROWD.exists(),
                                reason="crowd_world.json not generated yet")


@pytest.fixture
def env():
    return CrowdEnv(CROWD, seconds=20.0, seed=0, randomize_start=False)


def test_observation_shape_and_finiteness(env):
    o = env.reset()
    assert o.shape == (env.obs_dim,) == (7 + N_SLOTS * OBS_PER_SLOT,)
    assert np.all(np.isfinite(o))


def test_empty_slots_are_not_read_as_someone_at_the_origin(env):
    """A zero-filled slot would look like a person standing on the robot."""
    env.reset()
    o = env._obs()
    slots = o[7:].reshape(N_SLOTS, OBS_PER_SLOT)
    empty = slots[slots[:, 4] == 0.0]
    assert np.all(empty[:, 0] > 1.0), "empty slots must sit far ahead, not at 0"


def test_forward_action_moves_the_robot_forward(env):
    env.reset()
    p0 = env.pos.copy()
    env.step(np.array([1.0, 0.0, 0.0]))
    moved = env.pos - p0
    assert np.linalg.norm(moved) > 0.01
    fwd = np.array([np.cos(env.yaw), np.sin(env.yaw)])
    assert float(moved @ fwd) > 0


def test_backwards_command_never_reverses(env):
    env.reset()
    p0 = env.pos.copy()
    fwd = np.array([np.cos(env.yaw), np.sin(env.yaw)])
    env.step(np.array([-1.0, 0.0, 0.0]))
    assert float((env.pos - p0) @ fwd) >= -1e-9


def test_progress_toward_goal_is_rewarded(env):
    env.reset()
    _, r_fwd, *_ = env.step(np.array([1.0, 0.0, 0.0]))
    env.reset()
    _, r_still, *_ = env.step(np.array([0.0, 0.0, 0.0]))
    assert r_fwd > r_still


def test_crowd_positions_are_continuous_in_time(env):
    """Regression: wrapping crowd time teleported everyone back to their start
    at the period boundary, which read as a collision out of nowhere."""
    env.reset()
    worst = 0.0
    for tr in env.tracks:
        t0, t1 = float(tr["t"][0]), float(tr["t"][-1])
        ts = np.arange(t0, t1, 0.1)
        if len(ts) < 3:
            continue
        xs = np.interp(ts, tr["t"], tr["P"][:, 0])
        ys = np.interp(ts, tr["t"], tr["P"][:, 1])
        steps = np.hypot(np.diff(xs), np.diff(ys))
        worst = max(worst, float(steps.max()))
    assert worst < 0.45, f"a tracked person jumps {worst:.2f} m in one 0.1s step"


def test_crowd_is_present_through_an_episode(env):
    env.reset()
    counts = [len(env.crowd_at(t)) for t in np.arange(0.0, 12.0, 1.0)]
    assert min(counts) > 0, "the street must never be empty during an episode"


def test_episode_terminates_on_reaching_goal(env):
    env.reset()
    env.pos = env.goal_xy.astype(float) - np.array([0.5, 0.0])
    env.prev_goal_dist = 0.5
    _, r, done, _, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert info["reached"] and done and r > 10


def test_walking_out_of_the_street_ends_the_episode(env):
    env.reset()
    env.pos = np.array([env.pos[0], env.corridor[1] + 3.0])
    _, _, done, _, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert info["out_of_street"] and done


def test_reset_never_starts_inside_someone():
    """Regression: a random crowd phase spawned the robot already overlapping a
    pedestrian, so every episode began in collision and both controllers scored
    a 100% collision rate that had nothing to do with navigation."""
    import numpy as np
    e = CrowdEnv(CROWD, seconds=20.0, seed=3, randomize_start=True)
    for k in range(25):
        e.reset(seed=100 + k)
        crowd = e.crowd_at(e.t)
        if not crowd:
            continue
        clear = min(float(np.linalg.norm(o.pos - e.pos)) - o.radius
                    - e.p.robot_radius for o in crowd)
        assert clear > 0.0, f"episode {k} began already in collision"
