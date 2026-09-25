import numpy as np
import pytest

from densewalk import frames


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class TestPose:
    def test_identity_is_noop(self):
        p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = frames.Pose.identity().apply(p)
        assert np.allclose(out, p)

    def test_rejects_non_orthonormal(self):
        with pytest.raises(ValueError):
            frames.Pose(np.eye(3) * 2.0, np.zeros(3))

    def test_rejects_improper_rotation(self):
        R = np.eye(3)
        R[0, 0] = -1  # reflection, det = -1
        with pytest.raises(ValueError):
            frames.Pose(R, np.zeros(3))

    def test_inverse_round_trip(self):
        R = random_rotation(0)
        t = np.array([1.0, -2.0, 0.5])
        pose = frames.Pose(R, t)
        pts = np.random.default_rng(1).normal(size=(10, 3))
        out = pose.inverse().apply(pose.apply(pts))
        assert np.allclose(out, pts, atol=1e-9)

    def test_compose_matches_matrix_multiply(self):
        p1 = frames.Pose(random_rotation(2), np.array([1.0, 0.0, 0.0]))
        p2 = frames.Pose(random_rotation(3), np.array([0.0, 2.0, 0.0]))
        composed = p1.compose(p2)
        pt = np.array([[0.3, -0.4, 0.7]])
        expected = p1.apply(p2.apply(pt))
        assert np.allclose(composed.apply(pt), expected)

    def test_from_matrix_and_as_matrix_round_trip(self):
        pose = frames.Pose(random_rotation(4), np.array([5.0, -1.0, 2.0]))
        T = pose.as_matrix()
        pose2 = frames.Pose.from_matrix(T)
        assert np.allclose(pose.R, pose2.R)
        assert np.allclose(pose.t, pose2.t)


class TestAxisRemap:
    def test_identity_remap_is_noop(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(frames.IDENTITY_REMAP.apply(pts), pts)

    def test_yup_to_zup_moves_height_to_z(self):
        # Y-up point straight overhead: (0, h, 0) -> canonical (0, 0, h)
        pt = np.array([[0.0, 1.8, 0.0]])
        out = frames.YUP_TO_ZUP.apply(pt)
        assert np.allclose(out, [[0.0, 0.0, 1.8]])

    def test_yup_to_zup_is_orthonormal_and_invertible(self):
        M = frames.YUP_TO_ZUP.matrix()
        assert np.allclose(M.T @ M, np.eye(3))
        pts = np.random.default_rng(5).normal(size=(20, 3))
        pose = frames.YUP_TO_ZUP.as_pose()
        back = pose.inverse().apply(pose.apply(pts))
        assert np.allclose(back, pts, atol=1e-9)

    def test_bad_remap_rejected(self):
        bad = frames.AxisRemap(
            (frames.Axis.X, 1.0), (frames.Axis.X, 1.0), (frames.Axis.Z, 1.0)
        )  # X used twice -> not orthonormal
        with pytest.raises(ValueError):
            bad.matrix()


class TestUnitConversion:
    def test_cm_to_m_scalar(self):
        assert frames.cm_to_m(250.0) == pytest.approx(2.5)

    def test_cm_to_m_array(self):
        out = frames.cm_to_m(np.array([100.0, 200.0]))
        assert np.allclose(out, [1.0, 2.0])

    def test_mm_to_m(self):
        assert frames.mm_to_m(1800.0) == pytest.approx(1.8)


class TestCameraProjection:
    def _intr(self) -> frames.Intrinsics:
        return frames.Intrinsics(
            fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, width=1920, height=1080,
            dist=np.zeros(5),
        )

    def test_project_point_on_axis_lands_at_principal_point(self):
        intr = self._intr()
        world_to_cam = frames.Pose.identity()
        pt = np.array([[0.0, 0.0, 5.0]])  # straight ahead in camera frame
        pix = frames.world_to_pixel(pt, world_to_cam, intr)
        assert np.allclose(pix, [[960.0, 540.0]], atol=1e-6)

    def test_project_unproject_round_trip_direction(self):
        intr = self._intr()
        world_to_cam = frames.Pose.identity()
        pt_cam = np.array([[0.3, -0.2, 2.0]])
        pix = frames.world_to_pixel(pt_cam, world_to_cam, intr)
        ray = frames.pixel_to_ray_cam(pix, intr)[0]
        expected_dir = pt_cam[0] / np.linalg.norm(pt_cam[0])
        assert np.allclose(ray, expected_dir, atol=1e-4)

    def test_in_front_of_camera(self):
        world_to_cam = frames.Pose.identity()
        pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
        front = frames.is_in_front_of_camera(pts, world_to_cam)
        assert front.tolist() == [True, False]

    def test_opencv_rt_conversion_matches_manual_rodrigues(self):
        import cv2

        rvec = np.array([0.1, 0.2, 0.3])
        tvec = np.array([1.0, 2.0, 3.0])
        pose = frames.opencv_rt_to_world_to_cam_pose(rvec, tvec)
        R_expected, _ = cv2.Rodrigues(rvec)
        assert np.allclose(pose.R, R_expected)
        assert np.allclose(pose.t, tvec)


class TestSanityAsserts:
    def test_floor_assert_passes_near_zero(self):
        pts = np.array([[0.0, 0.0, 0.01], [1.0, 1.0, -0.02], [2.0, 0.0, 0.0]])
        frames.assert_floor_at_zero(pts, tol_m=0.05)  # should not raise

    def test_floor_assert_fails_when_offset(self):
        pts = np.array([[0.0, 0.0, 1.5], [1.0, 1.0, 1.4], [2.0, 0.0, 1.6]])
        with pytest.raises(AssertionError):
            frames.assert_floor_at_zero(pts, tol_m=0.05)

    def test_floor_assert_requires_min_points(self):
        with pytest.raises(ValueError):
            frames.assert_floor_at_zero(np.array([[0.0, 0.0, 0.0]]), min_points=3)

    def test_metric_scale_assert_passes(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])
        frames.assert_metric_scale(pts, expected_extent_m=(0.3, 2.5), axis=2)

    def test_metric_scale_assert_catches_missing_cm_conversion(self):
        # if someone forgot to convert cm->m, a 1.8m person becomes 180 "m"
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 180.0]])
        with pytest.raises(AssertionError):
            frames.assert_metric_scale(pts, expected_extent_m=(0.3, 2.5), axis=2)
