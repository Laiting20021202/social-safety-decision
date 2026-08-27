from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.ros2_bridge.edge_relationship import (
    EdgeMotionClassifier,
    EdgeTAMRelationshipBridge,
    edge_obstacle_array_to_track_states,
)
from realtime_safety.ros2_bridge.relationship_publisher import (
    build_relationship_payload,
)
from realtime_safety.types import RobotArmState


def _point(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _stamp(seconds: float) -> SimpleNamespace:
    whole = int(seconds)
    return SimpleNamespace(sec=whole, nanosec=round((seconds - whole) * 1e9))


def _obstacle(
    *,
    track_id: int = 7,
    state: str = "CONFIRMED",
    center: tuple[float, float, float] = (0.4, 0.8, 0.1),
    velocity: tuple[float, float, float] = (0.2, 0.0, 0.1),
    misses: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        tracking_state=state,
        filtered_centroid=_point(*center),
        velocity=_point(*velocity),
        aabb_min=_point(center[0] - 0.2, center[1] - 0.1, center[2] - 0.3),
        aabb_max=_point(center[0] + 0.2, center[1] + 0.1, center[2] + 0.3),
        uncertainty_margin=0.04,
        last_measurement_stamp=_stamp(12.25),
        hit_count=5,
        missed_frame_count=misses,
        confidence=0.9,
    )


def _array(*obstacles: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=_stamp(12.5),
            frame_id="realtime_safety_frame",
        ),
        obstacles=list(obstacles),
    )


def _robot() -> RobotArmState:
    return RobotArmState(
        center_xyz=np.array((0.1, 0.4, 0.2), dtype=np.float32),
        center_xy=np.array((100.0, 50.0), dtype=np.float32),
        image_size=(320, 240),
        mask_pixels=300,
        point_count=100,
        confidence=0.9,
        timestamp=12.5,
    )


def test_edge_conversion_preserves_id_and_converts_z_down_to_internal() -> None:
    converted = edge_obstacle_array_to_track_states(_array(_obstacle()))

    assert converted.timestamp == pytest.approx(12.5)
    assert converted.frame_id == "realtime_safety_frame"
    assert len(converted.tracks) == 1
    track = converted.tracks[0]
    assert track.track_id == 7
    assert track.class_name == "hand_candidate"
    np.testing.assert_allclose(track.position_xyz, (0.4, 0.8, -0.1))
    np.testing.assert_allclose(track.velocity_xyz, (0.2, 0.0, -0.1))
    np.testing.assert_allclose(track.bbox3d.minimum, (0.2, 0.7, -0.4))
    np.testing.assert_allclose(track.bbox3d.maximum, (0.6, 0.9, 0.2))
    assert track.radius == pytest.approx(np.hypot(0.2, 0.1))
    assert track.last_timestamp == pytest.approx(12.25)


def test_converted_edge_track_uses_existing_wire_contract_without_double_flip() -> None:
    track = edge_obstacle_array_to_track_states(_array(_obstacle())).tracks[0]
    payload = build_relationship_payload(
        _robot(),
        [track],
        coordinate_mode="camera_y_forward",
        source_timestamp=12.5,
    )

    obstacle = payload["obstacles"][0]
    assert obstacle["obstacle_center_m"] == pytest.approx(
        {"x": 0.4, "y": 0.8, "z": 0.1}
    )
    assert obstacle["velocity_mps"] == pytest.approx(
        {"x": 0.2, "y": 0.0, "z": 0.1}
    )
    assert obstacle["center_distance_m"] == pytest.approx(
        np.linalg.norm(np.array((0.4, 0.8, -0.1)) - _robot().center_xyz)
    )


def test_edge_conversion_accepts_rep103_optical_input() -> None:
    converted = edge_obstacle_array_to_track_states(
        _array(_obstacle(center=(0.4, -0.1, 0.8), velocity=(0.2, -0.1, 0.0))),
        source_coordinate_mode="ros_optical",
    )

    np.testing.assert_allclose(converted.tracks[0].position_xyz, (0.4, 0.8, 0.1))
    np.testing.assert_allclose(converted.tracks[0].velocity_xyz, (0.2, 0.0, 0.1))


def test_edge_conversion_rejects_tentative_but_keeps_occluded() -> None:
    converted = edge_obstacle_array_to_track_states(
        _array(
            _obstacle(track_id=1, state="TENTATIVE"),
            _obstacle(track_id=2, state="OCCLUDED", misses=2),
            _obstacle(track_id=3, state="DELETED"),
        )
    )

    assert [track.track_id for track in converted.tracks] == [2]
    assert converted.tracks[0].missing_count == 2


def test_edge_motion_classifier_matches_yolo_hysteresis() -> None:
    classifier = EdgeMotionClassifier(
        dynamic_enter_speed=0.15,
        dynamic_exit_speed=0.08,
        minimum_dynamic_hits=3,
    )

    assert [classifier.update(1, 0.2) for _ in range(3)] == [
        "static",
        "static",
        "dynamic",
    ]
    assert classifier.update(1, 0.10) == "dynamic"
    assert classifier.update(1, 0.05) == "dynamic"
    assert classifier.update(1, 0.05) == "dynamic"
    assert classifier.update(1, 0.05) == "static"


def test_bridge_only_forwards_active_backend_and_uses_latest_arm() -> None:
    published: list[tuple[RobotArmState | None, list, float]] = []

    class Publisher:
        def publish(self, robot_arm, tracks, *, source_timestamp):
            published.append((robot_arm, tracks, source_timestamp))

    bridge = EdgeTAMRelationshipBridge(Publisher(), initial_mode="yolo")
    bridge.update_robot_arm(_robot())
    assert not bridge.handle_message(_array(_obstacle()))

    bridge.set_mode("edgetam")
    # The old backend's arm center is intentionally cleared at the switch.
    bridge.update_robot_arm(_robot())
    assert bridge.handle_message(_array(_obstacle()))
    assert len(published) == 1
    assert published[0][0] is not None
    assert published[0][1][0].track_id == 7
    assert published[0][2] == pytest.approx(12.5)


def test_bridge_does_not_forward_edge_after_switching_to_yolo() -> None:
    published = []
    publisher = SimpleNamespace(
        publish=lambda robot_arm, tracks, *, source_timestamp: published.append(
            (robot_arm, tracks, source_timestamp)
        )
    )
    bridge = EdgeTAMRelationshipBridge(publisher, initial_mode="edgetam")

    assert bridge.handle_message(_array(_obstacle()))
    bridge.set_mode("yolo")
    assert not bridge.handle_message(_array(_obstacle()))
    assert len(published) == 1


def test_bridge_drops_stale_arm_center_but_keeps_obstacle_kinematics() -> None:
    published = []
    publisher = SimpleNamespace(
        publish=lambda robot_arm, tracks, *, source_timestamp: published.append(
            (robot_arm, tracks, source_timestamp)
        )
    )
    stale_arm = _robot()
    stale_arm.timestamp = 10.0
    bridge = EdgeTAMRelationshipBridge(
        publisher,
        initial_mode="edgetam",
        maximum_arm_age_sec=0.35,
    )
    bridge.update_robot_arm(stale_arm)

    assert bridge.handle_message(_array(_obstacle()))
    assert published[0][0] is None
    assert published[0][1][0].track_id == 7
    assert bridge.latest_frame is not None


def test_bridge_prefers_fresh_urdf_fk_arm_provider() -> None:
    published = []
    provider_stamps: list[float] = []
    publisher = SimpleNamespace(
        publish=lambda robot_arm, tracks, *, source_timestamp: published.append(
            (robot_arm, tracks, source_timestamp)
        )
    )
    fk_arm = _robot()
    fk_arm.localization_source = "urdf_fk_joint_state"

    def provide(timestamp: float) -> RobotArmState:
        provider_stamps.append(timestamp)
        fk_arm.timestamp = timestamp
        return fk_arm

    bridge = EdgeTAMRelationshipBridge(
        publisher,
        initial_mode="edgetam",
        robot_arm_provider=provide,
    )

    assert bridge.handle_message(_array(_obstacle()))
    assert provider_stamps == [pytest.approx(12.5)]
    assert published[0][0].localization_source == "urdf_fk_joint_state"


def test_bridge_drop_in_publish_only_accepts_yolo_while_yolo_is_active() -> None:
    published = []
    publisher = SimpleNamespace(
        publish=lambda robot_arm, tracks, *, source_timestamp: (
            published.append((robot_arm, tracks, source_timestamp)) or True
        )
    )
    bridge = EdgeTAMRelationshipBridge(publisher, initial_mode="edgetam")

    assert not bridge.publish(_robot(), [], source_timestamp=12.5)
    bridge.set_mode("yolo")
    assert bridge.publish(_robot(), [], source_timestamp=12.6)
    assert len(published) == 1
