import socket

import numpy as np
import viser

from realtime_safety.config import GuiConfig
from realtime_safety.gui.reconstruction_scene import (
    ReconstructionScene3D,
    _stable_horizontal_direction,
    _view_correction_quaternion,
)
from realtime_safety.gui.metric_bev import MetricBevCalibration
from realtime_safety.types import (
    BBox3D,
    PointCloudFrame,
    RobotArmState,
    Track3DState,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cloud(frame_index: int) -> PointCloudFrame:
    rng = np.random.default_rng(frame_index)
    points = rng.normal(size=(200, 3)).astype(np.float32) + np.array((4.0, 8.0, -2.0), np.float32)
    return PointCloudFrame(
        points=points,
        colors=np.full((len(points), 3), 128, np.uint8),
        confidence=np.ones(len(points), np.float32),
        pointmap=points.reshape(10, 20, 3),
        frame_index=frame_index,
        timestamp=float(frame_index),
        anchor_frame_index=0,
        inference_ms=1.0,
        valid=True,
        source="st4rtrack-test",
        tracking_points=points[:20],
    )


def test_reconstruction_history_is_centered_and_bounded() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(server, GuiConfig(history_frames=3))
    for frame_index in range(5):
        scene.update_pointcloud(_cloud(frame_index))
    assert scene.frame_count == 3
    assert scene.node_count == 10  # /frames plus three handles per retained timestep
    assert scene._center is not None
    assert np.linalg.norm(scene._center - np.array((4.0, 8.0, -2.0))) < 0.5
    center = np.array((4.0, 8.0, -2.0), dtype=np.float32)
    scene.update_people(
        4,
        [
            Track3DState(
                track_id=7,
                class_name="person",
                position_xyz=center,
                velocity_xyz=np.array((0.3, 0.1, 0.0), dtype=np.float32),
                acceleration_xyz=np.zeros(3, dtype=np.float32),
                covariance=np.eye(6),
                bbox3d=BBox3D(center - 0.2, center + 0.2),
                radius=0.3,
                hit_count=4,
                missing_count=0,
                last_timestamp=4.0,
                motion_state="dynamic",
                confidence=0.9,
                history=[
                    center + np.array((-0.09, -0.03, 0.0), dtype=np.float32),
                    center + np.array((-0.06, -0.02, 0.0), dtype=np.float32),
                    center + np.array((-0.03, -0.01, 0.0), dtype=np.float32),
                    center,
                ],
            )
        ],
    )
    assert scene.node_count == 14  # box, center, track label, and direction-arrow handle
    scene.reset()
    assert scene.frame_count == 0
    assert scene.node_count == 1
    scene.close()
    server.stop()


def test_direction_is_hidden_for_oscillating_person_centers() -> None:
    center = np.zeros(3, dtype=np.float32)
    track = Track3DState(
        track_id=3,
        class_name="person",
        position_xyz=center,
        velocity_xyz=np.array((4.0, 0.0, 0.0), dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(center - 0.2, center + 0.2),
        radius=0.3,
        hit_count=6,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="dynamic",
        confidence=0.9,
        history=[
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
        ],
    )
    assert _stable_horizontal_direction(track, np.full(3, 0.4, dtype=np.float32)) is None


def test_live_mode_reuses_one_persistent_webgl_pointcloud() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(server, GuiConfig(history_frames=1, history_stride=1))
    scene.update_aligned_frame(_cloud(0), [], yolo_count=0)
    first_handles = next(iter(scene._frames.values()))
    reconstruction_id = id(first_handles["reconstruction"])
    root_id = id(first_handles["root"])
    for frame_index in range(1, 8):
        scene.update_aligned_frame(_cloud(frame_index), [], yolo_count=0)
    current_handles = next(iter(scene._frames.values()))
    assert scene.frame_count == 1
    assert id(current_handles["reconstruction"]) == reconstruction_id
    assert id(current_handles["root"]) == root_id
    assert current_handles["root"].visible
    scene.close()
    server.stop()


def test_camera_mount_correction_rotates_every_visual_root_together() -> None:
    np.testing.assert_allclose(
        _view_correction_quaternion(0.0, 0.0, 0.0),
        np.array((1.0, 0.0, 0.0, 0.0)),
        atol=1e-12,
    )
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = ReconstructionScene3D(
        server, GuiConfig(history_frames=1, history_stride=1)
    )
    scene.set_view_correction(38.0, -3.0, 6.0)

    np.testing.assert_allclose(
        scene._frames_root.wxyz,
        np.array((1.0, 0.0, 0.0, 0.0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scene._edge_root.wxyz,
        np.array((1.0, 0.0, 0.0, 0.0)),
        atol=1e-12,
    )
    assert scene._bev_recalibrate_requested
    scene.close()
    server.stop()


def test_edge_obstacle_cloud_holds_one_frame_gap_then_clears() -> None:
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = ReconstructionScene3D(
        server, GuiConfig(history_frames=1, history_stride=1)
    )
    scene.update_pointcloud(_cloud(0))
    ros_points = np.array(
        [[4.0, 8.0, 2.0], [4.1, 8.0, 1.9]], dtype=np.float32
    )
    scene.update_edge_obstacle_cloud(ros_points, frame_id="realtime_safety_frame")

    assert scene._edge_obstacle_handle is not None
    expected = ros_points.copy()
    expected[:, 2] *= -1.0
    np.testing.assert_allclose(
        scene._edge_obstacle_handle.points,
        expected - scene._center,
        atol=2e-4,
    )
    assert "2 points" in scene._edge_foreground_status.content

    scene.update_edge_obstacle_cloud(np.empty((0, 3), dtype=np.float32))
    assert scene._edge_obstacle_handle is not None
    assert "HOLD" in scene._edge_foreground_status.content

    scene._edge_last_nonempty_at -= scene._edge_visual_hold_sec + 0.1
    scene.update_edge_obstacle_cloud(np.empty((0, 3), dtype=np.float32))
    assert scene._edge_obstacle_handle is None
    assert "0 points" in scene._edge_foreground_status.content
    scene.close()
    server.stop()


def test_simulator_edge_cloud_uses_declared_optical_frame_in_world() -> None:
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = ReconstructionScene3D(
        server, GuiConfig(history_frames=1, history_stride=1)
    )
    scene.configure_simulator_debug(
        {
            "zones": {"workspace": {"center": [0.0, 0.0, 0.0]}},
            "table": {
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 0.1],
                "yaw_deg": 0.0,
                "color_rgb": [0.1, 0.1, 0.1],
            },
            "apriltag": {"center": [0.0, 0.0, 0.06], "size": 0.08},
        },
        {
            "world_pose": {
                "position": [1.0, 2.0, 3.0],
                "rpy_deg": [0.0, 0.0, 0.0],
            }
        },
    )
    scene.update_edge_obstacle_cloud(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        frame_id="rgbd_color_optical_frame",
    )

    assert scene._edge_obstacle_handle is not None
    # ROS optical +Z maps to camera-link +X before the world pose.
    np.testing.assert_allclose(
        scene._edge_obstacle_handle.points, [[2.0, 2.0, 3.0]], atol=1e-6
    )
    scene.close()
    server.stop()


def test_simulator_world_cloud_replaces_gpu_handle_for_each_current_frame() -> None:
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = ReconstructionScene3D(
        server, GuiConfig(history_frames=1, history_stride=1)
    )
    scene.configure_simulator_debug(
        {
            "zones": {"workspace": {"center": [0.0, 0.0, 0.0]}},
            "table": {
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 0.1],
                "yaw_deg": 0.0,
                "color_rgb": [0.1, 0.1, 0.1],
            },
            "apriltag": {"center": [0.0, 0.0, 0.06], "size": 0.08},
        },
        {"world_pose": {"position": [0.0, 0.0, 1.0], "rpy_deg": [0.0, 0.0, 0.0]}},
    )
    scene._sim_render_interval_sec = 0.0
    first = scene._sim_debug_handles["world"]
    scene.update_simulator_debug_cloud(
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        np.array([[10, 20, 30]], dtype=np.uint8),
        True,
        15.0,
    )
    second = scene._sim_debug_handles["world"]
    assert second is not first
    np.testing.assert_allclose(second.points, [[0.1, 0.2, 0.3]], atol=1e-3)

    scene.update_simulator_debug_cloud(
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
        True,
        15.0,
    )
    third = scene._sim_debug_handles["world"]
    assert third is not second
    assert third.points.shape == (0, 3)
    assert "no history" in scene._sim_debug_status.content
    scene.close()
    server.stop()


def test_apriltag_task_plane_anchor_survives_temporary_tag_loss() -> None:
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = ReconstructionScene3D(
        server,
        GuiConfig(
            history_frames=1,
            history_stride=1,
            metric_bev_enabled=True,
        ),
    )
    calibration = MetricBevCalibration(
        origin=np.zeros(3, dtype=np.float32),
        right=np.array((1.0, 0.0, 0.0), dtype=np.float32),
        forward=np.array((0.0, 1.0, 0.0), dtype=np.float32),
        normal=np.array((0.0, 0.0, 1.0), dtype=np.float32),
        bounds_uv=(-0.8, 0.8, -0.8, 0.8),
        inlier_count=1000,
        inlier_ratio=0.9,
        rms_error_m=0.002,
    )
    fresh = _cloud(0)
    fresh.apriltag_locked = True
    fresh.apriltag_id = 0
    fresh.apriltag_size_m = 0.08
    fresh.apriltag_scale_correction = 0.5
    fresh.apriltag_corners_xyz = np.array(
        (
            (-0.04, 0.36, 0.0),
            (0.04, 0.36, 0.0),
            (0.04, 0.44, 0.0),
            (-0.04, 0.44, 0.0),
        ),
        dtype=np.float32,
    )
    scene._last_apriltag_scale = 0.5
    scene._bev_calibration = calibration
    scene._bev_recalibrate_requested = False
    scene._update_apriltag_locked(fresh, calibration)
    first_center = scene._apriltag_center_work.copy()
    first_corners = scene._apriltag_corners_work.copy()

    missing = _cloud(1)
    missing.apriltag_locked = False
    missing.apriltag_corners_xyz = None
    scene._update_apriltag_locked(missing, calibration)

    np.testing.assert_allclose(scene._apriltag_center_work, first_center)
    np.testing.assert_allclose(scene._apriltag_corners_work, first_corners)
    assert scene._bev_calibration is calibration
    assert scene._last_apriltag_scale == 0.5
    assert "HOLD" in scene._apriltag_handles["label"].text

    shifted = _cloud(2)
    shifted.apriltag_locked = True
    shifted.apriltag_id = 0
    shifted.apriltag_size_m = 0.08
    shifted.apriltag_scale_correction = 0.55
    shifted.apriltag_corners_xyz = fresh.apriltag_corners_xyz + np.array(
        (0.20, 0.15, 0.0), dtype=np.float32
    )
    scene._update_apriltag_locked(shifted, calibration)
    np.testing.assert_allclose(scene._apriltag_center_work, first_center)
    np.testing.assert_allclose(scene._apriltag_corners_work, first_corners)

    scene.recalibrate_metric_bev()
    assert scene._apriltag_center_work is None
    assert scene._apriltag_corners_work is None
    assert scene._last_apriltag_scale is None
    scene.close()
    server.stop()


def test_robot_and_obstacle_centers_have_visible_metric_relationship() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(
        server,
        GuiConfig(history_frames=1, presentation_mode=True),
    )
    cloud = _cloud(2)
    obstacle_center = np.array((4.3, 8.2, -1.9), dtype=np.float32)
    track = Track3DState(
        track_id=5,
        class_name="person",
        position_xyz=obstacle_center,
        velocity_xyz=np.zeros(3, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(obstacle_center - 0.2, obstacle_center + 0.2),
        radius=0.3,
        hit_count=5,
        missing_count=0,
        last_timestamp=2.0,
        motion_state="static",
        confidence=0.9,
    )
    robot = RobotArmState(
        center_xyz=np.array((4.0, 8.0, -2.0), dtype=np.float32),
        center_xy=np.array((160.0, 80.0), dtype=np.float32),
        image_size=(320, 240),
        mask_pixels=500,
        point_count=120,
        confidence=0.95,
        timestamp=2.0,
    )

    scene.update_aligned_frame(
        cloud,
        [track],
        yolo_count=1,
        robot_arm=robot,
    )

    handles = next(iter(scene._frames.values()))["people"]
    assert "robot:center" in handles
    assert "relation:lines" in handles
    assert "relation:label:5" in handles
    assert "0.37 m" in scene._relationship_status.content
    scene.close()
    server.stop()


def test_new_depth_frame_preserves_confirmed_geometry_until_yolo_catches_up() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(
        server,
        GuiConfig(history_frames=1, presentation_mode=True),
    )
    center = np.array((4.2, 8.1, -1.9), dtype=np.float32)
    track = Track3DState(
        track_id=12,
        class_name="person",
        position_xyz=center,
        velocity_xyz=np.zeros(3, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(center - 0.2, center + 0.2),
        radius=0.3,
        hit_count=4,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="static",
        confidence=0.9,
    )
    scene.update_aligned_frame(_cloud(1), [track], yolo_count=1)
    handles_before = next(iter(scene._frames.values()))["people"]
    box_handle = handles_before["box:12"]

    # The depth renderer may run before the matching YOLO worker. Updating the
    # persistent cloud must not remove the last confirmed obstacle.
    scene.update_pointcloud(_cloud(2))

    handles_after = next(iter(scene._frames.values()))["people"]
    assert handles_after["box:12"] is box_handle
    assert "center:12" in handles_after
    scene.close()
    server.stop()
