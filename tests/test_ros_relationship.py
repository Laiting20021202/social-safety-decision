from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.ros2_bridge.relationship_publisher import (
    ArmObstacleRelationshipPublisher,
    build_relationship_payload,
)
from realtime_safety.types import BBox3D, RobotArmState, Track3DState


def _robot() -> RobotArmState:
    return RobotArmState(
        center_xyz=np.array((0.1, 0.4, 0.2), dtype=np.float32),
        center_xy=np.array((160.0, 70.0), dtype=np.float32),
        image_size=(320, 240),
        mask_pixels=500,
        point_count=200,
        confidence=0.95,
        timestamp=1.0,
    )


def _track() -> Track3DState:
    center = np.array((0.4, 0.8, 0.1), dtype=np.float32)
    return Track3DState(
        track_id=7,
        class_name="person",
        position_xyz=center,
        velocity_xyz=np.array((0.2, 0.0, 0.1), dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(center - 0.2, center + 0.2),
        radius=0.15,
        hit_count=5,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="dynamic",
        confidence=0.9,
    )


def test_relationship_payload_contains_wire_centers_and_metric_distance() -> None:
    payload = build_relationship_payload(
        _robot(),
        [_track()],
        coordinate_mode="camera_y_forward",
        source_timestamp=12.5,
        sequence=3,
    )

    assert payload["schema_version"] == 1
    assert payload["sequence"] == 3
    assert payload["frame_id"] == "realtime_safety_frame"
    assert payload["coordinate_convention"] == "x_right_y_forward_z_down_m"
    assert payload["status"] == "tracking"
    assert payload["arm"]["center_m"] == pytest.approx(
        {"x": 0.1, "y": 0.4, "z": -0.2}
    )
    obstacle = payload["obstacles"][0]
    assert obstacle["track_id"] == 7
    assert obstacle["obstacle_center_m"] == pytest.approx(
        {"x": 0.4, "y": 0.8, "z": -0.1}
    )
    assert obstacle["delta_from_arm_m"] == pytest.approx(
        {"x": 0.3, "y": 0.4, "z": 0.1}
    )
    assert obstacle["center_distance_m"] == pytest.approx(
        np.sqrt(0.3**2 + 0.4**2 + 0.1**2)
    )
    assert payload["nearest_obstacle"]["track_id"] == 7


def test_relationship_payload_explicitly_reports_no_arm_or_obstacles() -> None:
    payload = build_relationship_payload(None, [])

    assert payload["status"] == "arm_not_localized"
    assert not payload["arm_valid"]
    assert payload["arm"] is None
    assert payload["obstacle_count"] == 0
    assert payload["obstacles"] == []


def test_relationship_payload_supports_rep103_optical_coordinates() -> None:
    payload = build_relationship_payload(
        _robot(),
        [_track()],
        coordinate_mode="ros_optical",
    )

    assert payload["coordinate_convention"] == "x_right_y_down_z_forward_m"
    assert payload["arm"]["center_m"] == pytest.approx(
        {"x": 0.1, "y": -0.2, "z": 0.4}
    )
    assert payload["obstacles"][0]["obstacle_center_m"] == pytest.approx(
        {"x": 0.4, "y": -0.1, "z": 0.8}
    )


def test_bimanual_relationship_uses_nearest_tcp_and_exports_both() -> None:
    robot = _robot()
    robot.localization_source = "urdf_fk_joint_state_bimanual"
    robot.link_points_xyz = {
        "left_tcp": np.array((0.35, 0.75, 0.10), dtype=np.float32),
        "right_tcp": np.array((-0.4, 0.4, 0.2), dtype=np.float32),
    }

    payload = build_relationship_payload(
        robot,
        [_track()],
        coordinate_mode="internal_z_up",
    )

    assert set(payload["arm"]["tcp_centers_m"]) == {"left_tcp", "right_tcp"}
    obstacle = payload["obstacles"][0]
    assert obstacle["nearest_arm_point"] == "left_tcp"
    assert obstacle["center_distance_m"] == pytest.approx(
        np.linalg.norm(_track().position_xyz - robot.link_points_xyz["left_tcp"])
    )
    assert payload["nearest_obstacle"]["nearest_arm_point"] == "left_tcp"


def test_relationship_publisher_emits_json_and_respects_rate_gate() -> None:
    class FakeString:
        def __init__(self) -> None:
            self.data = ""

    messages = []
    publisher = ArmObstacleRelationshipPublisher(max_rate_hz=10.0)
    publisher._node = SimpleNamespace()
    publisher._publisher = SimpleNamespace(publish=messages.append)
    publisher._string_type = FakeString

    assert publisher.publish(_robot(), [_track()], source_timestamp=2.0)
    assert not publisher.publish(_robot(), [_track()], source_timestamp=2.01)
    payload = json.loads(messages[0].data)
    assert payload["obstacle_count"] == 1
    assert payload["arm_valid"]


def test_relationship_topic_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="rate must be positive"):
        ArmObstacleRelationshipPublisher(max_rate_hz=0)
