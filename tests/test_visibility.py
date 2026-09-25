import numpy as np
import pytest

from densewalk import visibility as vis


def person(x, y, radius=0.25, height=1.8, pid=0):
    return vis.Person(id=pid, x=x, y=y, radius=radius, height=height)


class TestSegmentIntersectsCylinder:
    def test_person_directly_between_camera_and_point_occludes(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=5.0, y=0.0)  # sitting right on the line, halfway
        assert vis.segment_intersects_cylinder(cam, point, p) is True

    def test_person_off_to_the_side_does_not_occlude(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=5.0, y=5.0)  # 5m off the line, well outside radius
        assert vis.segment_intersects_cylinder(cam, point, p) is False

    def test_person_just_inside_radius_occludes(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=5.0, y=0.2, radius=0.25)  # 0.2m off line, radius 0.25
        assert vis.segment_intersects_cylinder(cam, point, p) is True

    def test_person_just_outside_radius_does_not_occlude(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=5.0, y=0.3, radius=0.25)  # 0.3m off line, radius 0.25
        assert vis.segment_intersects_cylinder(cam, point, p) is False

    def test_person_behind_camera_does_not_occlude(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=-5.0, y=0.0)  # behind the camera, not on the segment
        assert vis.segment_intersects_cylinder(cam, point, p) is False

    def test_person_beyond_point_does_not_occlude(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=15.0, y=0.0)  # past the background point
        assert vis.segment_intersects_cylinder(cam, point, p) is False

    def test_short_person_near_camera_does_not_occlude_far_floor_point(self):
        # camera is high up; a short cylinder right under the camera never
        # crosses the line's height at that horizontal position.
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        p = person(x=0.5, y=0.0, height=0.3)
        assert vis.segment_intersects_cylinder(cam, point, p) is False


class TestBackgroundPointVisible:
    def test_visible_with_no_people(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        assert vis.is_point_visible_from_camera(point, cam, []) is True

    def test_occluded_by_any_one_of_several_people(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        people = [person(x=2.0, y=5.0, pid=1), person(x=5.0, y=0.0, pid=2)]
        assert vis.is_point_visible_from_camera(point, cam, people) is False

    def test_visible_when_all_people_clear(self):
        cam = np.array([0.0, 0.0, 3.0])
        point = np.array([10.0, 0.0, 0.0])
        people = [person(x=2.0, y=5.0, pid=1), person(x=8.0, y=-5.0, pid=2)]
        assert vis.is_point_visible_from_camera(point, cam, people) is True


class TestFrameVisibleAnyCamera:
    def test_visible_if_at_least_one_camera_clear(self):
        point = np.array([10.0, 0.0, 0.0])
        cams = [np.array([0.0, 0.0, 3.0]), np.array([10.0, -8.0, 3.0])]
        # occludes camera 0's view but not camera 1's (person near cam0's line)
        people = [person(x=5.0, y=0.0, pid=1)]
        assert vis.is_point_visible_any_camera(point, cams, people) is True

    def test_occluded_if_all_cameras_blocked(self):
        point = np.array([10.0, 0.0, 0.0])
        cams = [np.array([0.0, 0.0, 3.0])]
        people = [person(x=5.0, y=0.0, pid=1)]
        assert vis.is_point_visible_any_camera(point, cams, people) is False


class TestStaticVisibilityFraction:
    def test_fully_visible_scene_has_fraction_one(self):
        points = np.array([[10.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
        cams = [np.array([0.0, 0.0, 3.0])]
        frames = [vis.FrameOccupants(people=[]) for _ in range(4)]
        frac = vis.static_visibility_fraction(points, cams, frames)
        assert np.allclose(frac, [1.0, 1.0])

    def test_partially_occluded_fraction_matches_occlusion_count(self):
        point = np.array([[10.0, 0.0, 0.0]])
        cams = [np.array([0.0, 0.0, 3.0])]
        # occluded in 2 of 5 frames, clear in the other 3
        frames = [
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),  # occluded
            vis.FrameOccupants(people=[]),
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),  # occluded
            vis.FrameOccupants(people=[]),
            vis.FrameOccupants(people=[]),
        ]
        frac = vis.static_visibility_fraction(point, cams, frames)
        assert frac.shape == (1,)
        assert frac[0] == pytest.approx(3.0 / 5.0)

    def test_per_point_independence(self):
        points = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        cams = [np.array([0.0, 0.0, 3.0])]
        # this person only occludes the first point's line of sight
        frames = [vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)])]
        frac = vis.static_visibility_fraction(points, cams, frames)
        assert frac[0] == pytest.approx(0.0)
        assert frac[1] == pytest.approx(1.0)

    def test_rejects_empty_frames(self):
        points = np.array([[10.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            vis.static_visibility_fraction(points, [np.zeros(3)], [])


class TestBatchMatchesScalarReference:
    def test_matches_on_random_scenes(self):
        rng = np.random.default_rng(42)
        points = np.concatenate(
            [rng.uniform(-5, 15, size=(15, 2)), np.zeros((15, 1))], axis=1
        )
        cams = [np.array([rng.uniform(-2, 14), rng.uniform(-2, 40), rng.uniform(2, 5)]) for _ in range(3)]
        frames = []
        for _ in range(8):
            n_people = rng.integers(0, 6)
            people = [
                vis.Person(
                    id=i,
                    x=rng.uniform(-3, 17),
                    y=rng.uniform(-3, 42),
                    radius=rng.uniform(0.15, 0.35),
                    height=rng.uniform(1.4, 2.0),
                )
                for i in range(n_people)
            ]
            frames.append(vis.FrameOccupants(people=people))

        scalar = vis.static_visibility_fraction(points, cams, frames)
        batch = vis.static_visibility_fraction_batch(points, cams, frames)
        assert np.allclose(scalar, batch), (scalar, batch)

    def test_matches_on_known_fixture_from_earlier_tests(self):
        point = np.array([[10.0, 0.0, 0.0]])
        cams = [np.array([0.0, 0.0, 3.0])]
        frames = [
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),
            vis.FrameOccupants(people=[]),
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),
            vis.FrameOccupants(people=[]),
            vis.FrameOccupants(people=[]),
        ]
        scalar = vis.static_visibility_fraction(point, cams, frames)
        batch = vis.static_visibility_fraction_batch(point, cams, frames)
        assert np.allclose(scalar, batch)


class TestViewLevelVisibility:
    """'share of frames visible from >=1 camera' (frame-level, coarse:
    at most F+1 distinct values) is a different quantity from 'share of
    (camera,frame) observations that are unoccluded' (view-level, up to
    C*F distinct values). Only the joint version gives usable resolution
    for stratification."""

    def test_view_level_has_finer_resolution_than_frame_level(self):
        point = np.array([[10.0, 0.0, 0.0]])
        cam_blocked = np.array([0.0, 0.0, 3.0])  # person sits on this cam's line
        cam_clear = np.array([10.0, -20.0, 3.0])  # far off to the side, always clear
        cams = [cam_blocked, cam_clear]
        frames = [
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),
            vis.FrameOccupants(people=[person(x=5.0, y=0.0, pid=1)]),
        ]

        frame_level = vis.static_visibility_fraction_batch(point, cams, frames)
        view_level = vis.static_visibility_fraction_view_level_batch(point, cams, frames)

        # frame-level: cam_clear always saves it -> every frame counts visible
        assert frame_level[0] == pytest.approx(1.0)
        # view-level: 1 of 2 cameras occluded in both frames -> 2/4 pairs visible
        assert view_level[0] == pytest.approx(0.5)
        assert view_level[0] != frame_level[0]

    def test_view_level_matches_brute_force(self):
        rng = np.random.default_rng(7)
        points = np.concatenate(
            [rng.uniform(-5, 15, size=(10, 2)), np.zeros((10, 1))], axis=1
        )
        cams = [np.array([rng.uniform(-2, 14), rng.uniform(-2, 40), rng.uniform(2, 5)]) for _ in range(4)]
        frames = []
        for _ in range(6):
            n_people = rng.integers(0, 5)
            people = [
                vis.Person(
                    id=i, x=rng.uniform(-3, 17), y=rng.uniform(-3, 42),
                    radius=rng.uniform(0.15, 0.35), height=rng.uniform(1.4, 2.0),
                )
                for i in range(n_people)
            ]
            frames.append(vis.FrameOccupants(people=people))

        view_level = vis.static_visibility_fraction_view_level_batch(points, cams, frames)

        n = points.shape[0]
        expected = np.zeros(n)
        total_pairs = len(cams) * len(frames)
        for frame in frames:
            for cam in cams:
                for i in range(n):
                    if vis.is_point_visible_from_camera(points[i], cam, frame.people):
                        expected[i] += 1.0
        expected /= total_pairs
        assert np.allclose(view_level, expected)

    def test_no_people_gives_fraction_one(self):
        points = np.array([[10.0, 0.0, 0.0]])
        cams = [np.array([0.0, 0.0, 3.0]), np.array([5.0, 5.0, 3.0])]
        frames = [vis.FrameOccupants(people=[]) for _ in range(3)]
        out = vis.static_visibility_fraction_view_level_batch(points, cams, frames)
        assert np.allclose(out, [1.0])
