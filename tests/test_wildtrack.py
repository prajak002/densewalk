import numpy as np
import pytest
import cv2

from densewalk import wildtrack as wt


class TestPositionGridToWorld:
    def test_id_zero_is_grid_origin(self):
        # per the dataset's own README.txt:
        #   X = -3.0 + 0.025*(ID%480), Y = -9.0 + 0.025*(ID//480)
        pts = wt.position_id_to_world(np.array([0]))
        assert pts.shape == (1, 3)
        assert np.allclose(pts[0], [-3.0, -9.0, 0.0])

    def test_id_one_steps_along_x(self):
        # i % 480 controls the x index per the toolkit's project_grid_points
        pts = wt.position_id_to_world(np.array([0, 1]))
        step_m = pts[1, 0] - pts[0, 0]
        assert step_m == pytest.approx(0.025)  # 2.5cm step
        assert pts[1, 1] == pytest.approx(pts[0, 1])  # y unchanged

    def test_id_480_steps_along_y(self):
        pts = wt.position_id_to_world(np.array([0, 480]))
        assert pts[1, 0] == pytest.approx(pts[0, 0])  # x wraps back
        step_m = pts[1, 1] - pts[0, 1]
        assert step_m == pytest.approx(0.025)

    def test_all_points_on_floor(self):
        ids = np.array([0, 479, 480, 691199])
        pts = wt.position_id_to_world(ids)
        assert np.allclose(pts[:, 2], 0.0)

    def test_max_id_matches_grid_extent(self):
        max_id = wt.GRID_WIDTH * wt.GRID_HEIGHT - 1
        pts = wt.position_id_to_world(np.array([max_id]))
        expected_x = -3.0 + 0.025 * (wt.GRID_WIDTH - 1)
        expected_y = -9.0 + 0.025 * (wt.GRID_HEIGHT - 1)
        assert pts[0, 0] == pytest.approx(expected_x)
        assert pts[0, 1] == pytest.approx(expected_y)

    def test_grid_spans_the_documented_plaza_extent(self):
        # 480 x 1440 cells at 2.5cm => 12m x 36m, matching the paper
        corners = wt.position_id_to_world(
            np.array([0, wt.GRID_WIDTH - 1, wt.GRID_WIDTH * (wt.GRID_HEIGHT - 1)])
        )
        assert corners[1, 0] - corners[0, 0] == pytest.approx(11.975)
        assert corners[2, 1] - corners[0, 1] == pytest.approx(35.975)

    def test_rejects_out_of_range_id(self):
        with pytest.raises(ValueError):
            wt.position_id_to_world(np.array([-1]))
        with pytest.raises(ValueError):
            wt.position_id_to_world(np.array([wt.GRID_WIDTH * wt.GRID_HEIGHT]))


class TestOpenCvXmlRoundTrip:
    def test_read_back_matrix_written_by_opencv(self, tmp_path):
        path = str(tmp_path / "calib.xml")
        mat = np.array([[1700.0, 0.0, 960.0], [0.0, 1700.0, 540.0], [0.0, 0.0, 1.0]])
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", mat)
        fs.release()

        out = wt.load_opencv_xml_matrix(path, "camera_matrix")
        assert np.allclose(out, mat)

    def test_read_back_vector(self, tmp_path):
        path = str(tmp_path / "extr.xml")
        rvec = np.array([[0.1], [0.2], [0.3]])
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("rvec", rvec)
        fs.release()

        out = wt.load_opencv_xml_matrix(path, "rvec")
        assert np.allclose(out.reshape(-1), rvec.reshape(-1))

    def test_missing_tag_raises(self, tmp_path):
        path = str(tmp_path / "empty.xml")
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("something_else", np.eye(3))
        fs.release()
        with pytest.raises(KeyError):
            wt.load_opencv_xml_matrix(path, "camera_matrix")


WILDTRACK_EXTRINSIC_XML = """<?xml version="1.0"?>
<opencv_storage>
  <rvec>
    1.759099006652832 0.46710100769996643 -0.331699013710022
   </rvec>
  <tvec>
    -525.8941650390625 45.40763473510742 986.7235107421875
  </tvec>
</opencv_storage>
"""


class TestPlainTextVectorNodes:
    """WildTrack's extrinsic files store rvec/tvec as bare whitespace-
    separated text, NOT as type_id='opencv-matrix' nodes, so
    cv2.FileStorage(...).mat() returns nothing for them."""

    def test_reads_plain_text_rvec(self, tmp_path):
        path = tmp_path / "extr.xml"
        path.write_text(WILDTRACK_EXTRINSIC_XML)
        rvec = wt.load_opencv_xml_matrix(str(path), "rvec")
        assert rvec.reshape(-1).shape == (3,)
        assert rvec.reshape(-1)[0] == pytest.approx(1.759099006652832)
        assert rvec.reshape(-1)[2] == pytest.approx(-0.331699013710022)

    def test_reads_plain_text_tvec(self, tmp_path):
        path = tmp_path / "extr.xml"
        path.write_text(WILDTRACK_EXTRINSIC_XML)
        tvec = wt.load_opencv_xml_matrix(str(path), "tvec")
        assert tvec.reshape(-1)[2] == pytest.approx(986.7235107421875)

    def test_missing_tag_still_raises_on_plain_text_file(self, tmp_path):
        path = tmp_path / "extr.xml"
        path.write_text(WILDTRACK_EXTRINSIC_XML)
        with pytest.raises(KeyError):
            wt.load_opencv_xml_matrix(str(path), "camera_matrix")


class TestTvecUnits:
    """tvec is in centimetres in this dataset (|t| ~ 1000 => ~10m);
    the canonical frame is metres."""

    def test_load_camera_converts_tvec_to_metres(self, tmp_path):
        extr = tmp_path / "extr.xml"
        extr.write_text(WILDTRACK_EXTRINSIC_XML)
        intr_path = tmp_path / "intr.xml"
        K = np.array([[1743.4, 0.0, 934.5], [0.0, 1735.2, 444.4], [0.0, 0.0, 1.0]])
        fs = cv2.FileStorage(str(intr_path), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", np.zeros((5, 1)))
        fs.release()

        intr, world_to_cam = wt.load_camera(str(intr_path), str(extr), 1920, 1080)
        # translation magnitude must be metres-scale, not centimetres-scale
        assert np.linalg.norm(world_to_cam.t) < 50.0
        assert world_to_cam.t[2] == pytest.approx(9.867235107421875, abs=1e-6)

    def test_loaded_camera_height_is_plausible(self, tmp_path):
        extr = tmp_path / "extr.xml"
        extr.write_text(WILDTRACK_EXTRINSIC_XML)
        intr_path = tmp_path / "intr.xml"
        K = np.array([[1743.4, 0.0, 934.5], [0.0, 1735.2, 444.4], [0.0, 0.0, 1.0]])
        fs = cv2.FileStorage(str(intr_path), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", np.zeros((5, 1)))
        fs.release()
        _, world_to_cam = wt.load_camera(str(intr_path), str(extr), 1920, 1080)
        wt.assert_camera_pose_plausible(world_to_cam)  # must not raise


class TestLoadCameraPose:
    def test_extrinsic_intrinsic_compose_into_world_to_cam_pose(self, tmp_path):
        intr_path = str(tmp_path / "intr.xml")
        extr_path = str(tmp_path / "extr.xml")

        K = np.array([[1700.0, 0.0, 960.0], [0.0, 1700.0, 540.0], [0.0, 0.0, 1.0]])
        dist = np.zeros((1, 5))
        fs = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.release()

        rvec = np.zeros((3, 1))
        tvec = np.array([[0.0], [0.0], [300.0]])  # centimetres, as in the dataset
        fs = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_WRITE)
        fs.write("rvec", rvec)
        fs.write("tvec", tvec)
        fs.release()

        intr, world_to_cam = wt.load_camera(intr_path, extr_path, width=1920, height=1080)
        assert intr.fx == pytest.approx(1700.0)
        # 300cm -> 3m; world_to_cam maps world origin to (0,0,3) in the
        # camera frame under identity rotation
        assert np.allclose(world_to_cam.apply(np.zeros((1, 3))), [[0.0, 0.0, 3.0]])
