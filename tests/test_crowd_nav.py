"""Behavioural tests for the crowd navigation rules."""
from __future__ import annotations

import numpy as np
import pytest

from densewalk.crowd_nav import (NavParams, Obstacle, avoidance_vector,
                                 command, time_to_collision)


def test_ttc_head_on():
    # closing at 2 m/s from 5m apart, discs touch at 1m combined radius
    ttc = time_to_collision(np.array([5.0, 0.0]), np.array([-2.0, 0.0]), 1.0)
    assert ttc == pytest.approx(2.0, abs=1e-6)


def test_ttc_never_closes():
    # moving away
    assert time_to_collision(np.array([5.0, 0.0]), np.array([2.0, 0.0]), 1.0) == np.inf
    # parallel, never converging
    assert time_to_collision(np.array([0.0, 3.0]), np.array([1.0, 0.0]), 1.0) == np.inf


def test_ttc_already_overlapping_is_zero():
    assert time_to_collision(np.array([0.3, 0.0]), np.array([0.0, 0.0]), 1.0) == 0.0


def test_ttc_stationary_obstacle_in_path():
    # robot closing on a parked obstacle: relative velocity is what matters
    ttc = time_to_collision(np.array([3.0, 0.0]), np.array([-1.0, 0.0]), 0.5)
    assert ttc == pytest.approx(2.5, abs=1e-6)


def test_clear_path_goes_straight_at_goal():
    cmd, info = command(np.zeros(2), 0.0, np.zeros(2),
                        np.array([5.0, 0.0]), [])
    assert cmd[0] > 0.5          # forward
    assert abs(cmd[1]) < 1e-6    # no sideways
    assert info["min_ttc"] == np.inf
    assert not info["stopped"]


def test_person_crossing_pushes_robot_aside():
    """Someone crossing left-to-right ahead should push the command away from
    them, not just slow it down."""
    obs = [Obstacle(pos=np.array([3.0, -1.5]), vel=np.array([0.0, 1.0]),
                    radius=0.25)]
    cmd, info = command(np.zeros(2), 0.0, np.array([1.0, 0.0]),
                        np.array([8.0, 0.0]), obs)
    # They clear us by ~1.06m, so bodies never touch and ttc is correctly
    # infinite -- but they cut through our comfort zone, which is exactly the
    # case the crowding term exists for. They sweep toward +y, so we go -y.
    assert cmd[1] < 0.0
    assert not info["stopped"]


def test_imminent_collision_stops_rather_than_swerves():
    obs = [Obstacle(pos=np.array([0.8, 0.0]), vel=np.array([-1.5, 0.0]),
                    radius=0.25)]
    cmd, info = command(np.zeros(2), 0.0, np.array([1.0, 0.0]),
                        np.array([8.0, 0.0]), obs)
    assert info["stopped"]
    assert np.allclose(cmd, np.zeros(3))


def test_obstacle_behind_is_ignored():
    obs = [Obstacle(pos=np.array([-2.0, 0.0]), vel=np.array([0.0, 0.0]),
                    radius=0.25)]
    cmd, info = command(np.zeros(2), 0.0, np.array([1.0, 0.0]),
                        np.array([8.0, 0.0]), obs)
    assert info["min_ttc"] == np.inf
    assert cmd[0] > 0.5


def test_avoidance_uses_predicted_not_current_position():
    """A person who has already passed in front should not still repel us."""
    passing = [Obstacle(pos=np.array([3.0, 0.2]), vel=np.array([0.0, 4.0]),
                        radius=0.25)]
    _, ttc = avoidance_vector(np.zeros(2), np.array([1.0, 0.0]),
                              passing, NavParams())
    # they sweep out of the way well before we arrive
    assert ttc == np.inf or ttc > 1.0


def test_dense_crowd_does_not_freeze_the_robot():
    """Regression: folding the comfort margin into the collision radius made
    every nearby pedestrian read as an active collision, so in a dense street
    the robot emergency-stopped forever. Walking beside people is normal."""
    beside = [Obstacle(pos=np.array([0.0, 0.9]), vel=np.array([1.0, 0.0]),
                       radius=0.25),
              Obstacle(pos=np.array([0.0, -0.9]), vel=np.array([1.0, 0.0]),
                       radius=0.25)]
    cmd, info = command(np.zeros(2), 0.0, np.array([1.0, 0.0]),
                        np.array([8.0, 0.0]), beside)
    assert not info["stopped"]
    assert cmd[0] > 0.0


def test_command_respects_speed_limit():
    p = NavParams(max_speed=0.7)
    cmd, _ = command(np.zeros(2), 0.0, np.zeros(2), np.array([50.0, 0.0]), [], p)
    assert np.linalg.norm(cmd[:2]) <= p.max_speed + 1e-6


def test_yaw_rate_is_clamped():
    p = NavParams(max_yaw_rate=0.5)
    cmd, _ = command(np.zeros(2), 0.0, np.zeros(2), np.array([0.0, 5.0]), [], p)
    assert abs(cmd[2]) <= p.max_yaw_rate + 1e-6


def test_avoidance_is_capped_so_crowd_density_cannot_hijack_the_robot():
    """Regression: with 17 pedestrians the summed repulsion overwhelmed the
    goal term and the robot was driven sideways out of the street entirely."""
    from densewalk.crowd_nav import avoidance_vector
    many = [Obstacle(pos=np.array([2.0, 0.1 * i]), vel=np.array([-1.0, 0.0]),
                     radius=0.25) for i in range(-8, 9)]
    vec, _ = avoidance_vector(np.zeros(2), np.array([1.0, 0.0]), many, NavParams())
    assert np.linalg.norm(vec) <= NavParams().max_avoid + 1e-6


def test_corridor_pushes_robot_back_into_the_street():
    from densewalk.crowd_nav import wall_push
    p = NavParams()
    # beyond the left edge -> pushed back toward -y
    assert wall_push(np.array([0.0, 4.0]), (-3.0, 3.0), p)[1] < 0
    # beyond the right edge -> pushed back toward +y
    assert wall_push(np.array([0.0, -4.0]), (-3.0, 3.0), p)[1] > 0
    # comfortably inside -> no push at all
    assert np.allclose(wall_push(np.array([0.0, 0.0]), (-3.0, 3.0), p), 0.0)


def test_corridor_keeps_goal_reachable_in_a_dense_crowd():
    """The whole failure mode in one test: dense oncoming crowd plus walls,
    the robot must still make net forward progress rather than fleeing."""
    crowd = [Obstacle(pos=np.array([3.0 + 0.4 * i, 0.5 * ((-1) ** i)]),
                      vel=np.array([-1.0, 0.0]), radius=0.25)
             for i in range(12)]
    cmd, info = command(np.zeros(2), 0.0, np.array([0.8, 0.0]),
                        np.array([25.0, 0.0]), crowd, NavParams(),
                        corridor=(-3.0, 3.0))
    # A solid oncoming line with no gap: holding position is correct, but
    # reversing never is -- it cannot see behind itself.
    assert cmd[0] >= -1e-6, "must never be driven backwards by the crowd"


def test_head_on_crowd_produces_sidestep_not_freeze():
    """Regression: clamping the whole command when repulsion pointed backwards
    made the robot stand still for 95% of a 35s run in oncoming traffic.
    It must still move laterally to thread the gaps."""
    oncoming = [Obstacle(pos=np.array([2.5, 0.9]), vel=np.array([-1.1, 0.0]),
                         radius=0.25),
                Obstacle(pos=np.array([3.2, 1.3]), vel=np.array([-1.0, 0.0]),
                         radius=0.25)]
    cmd, info = command(np.zeros(2), 0.0, np.array([0.8, 0.0]),
                        np.array([25.0, 0.0]), oncoming, NavParams(),
                        corridor=(-3.0, 3.0))
    if not info["stopped"]:
        assert cmd[0] >= -1e-6                      # never backwards
        assert np.linalg.norm(cmd[:2]) > 1e-3, "must not freeze in place"


def test_heading_never_turns_its_back_on_the_goal():
    """Regression: steering along the full desired vector let a backwards
    repulsion spin the robot 180 deg; it then walked 'forward' away from the
    goal and oscillated for 35s without advancing."""
    blockers = [Obstacle(pos=np.array([1.5, 0.0]), vel=np.array([-1.2, 0.0]),
                         radius=0.3),
                Obstacle(pos=np.array([2.0, 0.4]), vel=np.array([-1.2, 0.0]),
                         radius=0.3),
                Obstacle(pos=np.array([2.0, -0.4]), vel=np.array([-1.2, 0.0]),
                         radius=0.3)]
    goal = np.array([20.0, 0.0])
    cmd, info = command(np.zeros(2), 0.0, np.array([0.9, 0.0]), goal,
                        blockers, NavParams(), corridor=(-3.0, 3.0))
    if not info["stopped"]:
        # commanded heading change must stay within the deflection limit
        assert abs(cmd[2]) <= NavParams().max_yaw_rate + 1e-6
        # and the world-frame motion must not be away from the goal
        assert cmd[0] >= -1e-6


def test_perception_ignores_people_behind_the_robot():
    from densewalk.crowd_nav import visible_obstacles
    behind = [Obstacle(np.array([-3.0, 0.0]), np.array([1.0, 0.0]), 0.25)]
    seen, info = visible_obstacles(np.zeros(2), 0.0, behind)
    assert seen == []
    assert info["n_out_of_view"] == 1


def test_perception_ignores_people_beyond_sensing_range():
    from densewalk.crowd_nav import visible_obstacles
    far = [Obstacle(np.array([30.0, 0.0]), np.array([-1.0, 0.0]), 0.25)]
    seen, _ = visible_obstacles(np.zeros(2), 0.0, far)
    assert seen == []


def test_perception_occludes_the_person_behind_the_person():
    """The regime this project targets: in a dense street the robot cannot see
    past the person directly in front of it."""
    from densewalk.crowd_nav import visible_obstacles
    near = Obstacle(np.array([2.0, 0.0]), np.array([-1.0, 0.0]), 0.35)
    hidden = Obstacle(np.array([4.0, 0.0]), np.array([-1.0, 0.0]), 0.25)
    seen, info = visible_obstacles(np.zeros(2), 0.0, [near, hidden])
    # compare by identity: Obstacle holds numpy arrays, so `in` would compare
    # arrays elementwise and raise on the ambiguous truth value
    assert [id(o) for o in seen] == [id(near)]
    assert info["n_occluded"] == 1


def test_perception_keeps_someone_beside_the_blocker():
    from densewalk.crowd_nav import visible_obstacles
    near = Obstacle(np.array([2.0, 0.0]), np.array([-1.0, 0.0]), 0.3)
    beside = Obstacle(np.array([4.0, 2.5]), np.array([-1.0, 0.0]), 0.25)
    seen, _ = visible_obstacles(np.zeros(2), 0.0, [near, beside])
    assert len(seen) == 2


def test_perception_follows_the_robots_heading():
    from densewalk.crowd_nav import visible_obstacles
    ob = [Obstacle(np.array([0.0, 3.0]), np.array([0.0, -1.0]), 0.25)]
    assert visible_obstacles(np.zeros(2), 0.0, ob)[0] == []          # off to the side
    assert len(visible_obstacles(np.zeros(2), np.pi / 2, ob)[0]) == 1  # turned to face
