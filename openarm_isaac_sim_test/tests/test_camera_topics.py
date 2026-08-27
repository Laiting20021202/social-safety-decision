from __future__ import annotations

import numpy as np
import pytest

from openarm_sim.camera_math import (
    back_project_depth,
    camera_aim_direction,
    camera_world_position,
    intrinsics_from_horizontal_fov,
    optical_points_to_world,
)
from openarm_sim.config import load_yaml
from openarm_sim.contracts import CAMERA_TOPICS


def test_camera_topic_contract_and_depth_units() -> None:
    camera = load_yaml("config/camera.yaml")["camera"]
    assert camera["depth_encoding"] == "32FC1"
    assert camera["depth_unit"] == "meter"
    # Native Gazebo topics remain part of the sensor contract, but production
    # GUI/avoidance must consume the RGB-D reconstruction topics below.
    assert set(CAMERA_TOPICS).issubset(set(camera["topics"].values()))
    assert camera["topics"]["reconstructed_points"] == "/realtime_safety/pointcloud"
    assert (
        camera["topics"]["reconstructed_world_points"]
        == "/realtime_safety/environment_cloud_world"
    )
    noise = camera["noise"]
    assert isinstance(noise["enabled"], bool)
    assert 0.0 <= float(noise["depth_stddev_m"]) <= 0.005


def test_explicit_camera_pose_matches_calibrated_mount_fields() -> None:
    camera = load_yaml("config/camera.yaml")["camera"]
    scene = load_yaml("config/scene.yaml")
    pose = camera["world_pose"]
    position = np.asarray(pose["position"], dtype=float)
    pitch = np.radians(float(pose["rpy_deg"][1]))
    direction = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    tilt_from_vertical = np.degrees(np.arccos(np.clip(-direction[2], -1.0, 1.0)))
    assert camera["tilt_reference"] == "vertical_down"
    assert position[2] - scene["table"]["top_z"] == pytest.approx(
        camera["height_above_table"]
    )
    assert tilt_from_vertical == pytest.approx(camera["tilt_from_vertical_deg"], abs=0.1)


def test_optical_point_transforms_to_world_metric_scale() -> None:
    points = optical_points_to_world(
        np.array([[0.0, 0.0, 1.0], [0.08, 0.0, 1.0]], dtype=np.float32),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert points[0] == pytest.approx([2.0, 2.0, 3.0])
    assert np.linalg.norm(points[1] - points[0]) == pytest.approx(0.08)


def test_aligned_depth_back_projection() -> None:
    intrinsics = intrinsics_from_horizontal_fov(4, 2, 90.0)
    depth = np.ones((2, 4), dtype=np.float32)
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    points, colors = back_project_depth(depth, intrinsics, rgb, 0.1, 3.0)
    assert points.shape == (8, 3)
    assert colors is not None and colors.shape == (8, 3)
    np.testing.assert_allclose(points[:, 2], 1.0)
