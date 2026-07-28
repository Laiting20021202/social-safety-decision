from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np

from realtime_safety.types import RobotArmState, Track3DState

LOGGER = logging.getLogger(__name__)


def _coordinate_xyz(
    values: np.ndarray,
    coordinate_mode: str,
) -> np.ndarray:
    xyz = np.asarray(values, dtype=np.float32).reshape(3).copy()
    if coordinate_mode == "camera_y_forward":
        # Internal reconstruction is x-right/y-forward/z-up. Koch VAMP uses
        # x-right/y-forward/z-down.
        xyz[2] *= -1.0
    return xyz


def _xyz_object(values: np.ndarray) -> dict[str, float]:
    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
    }


def build_relationship_payload(
    robot_arm: RobotArmState | None,
    tracks: list[Track3DState],
    *,
    frame_id: str = "realtime_safety_frame",
    coordinate_mode: str = "camera_y_forward",
    source_timestamp: float | None = None,
    sequence: int = 0,
    perception_sequence: int = 0,
    perception_age_sec: float = 0.0,
) -> dict[str, Any]:
    """Build the stable, versioned JSON contract consumed by the Koch NUC."""

    if coordinate_mode not in {"internal_z_up", "camera_y_forward"}:
        raise ValueError(f"Unsupported relationship coordinate mode: {coordinate_mode}")

    arm_valid = bool(
        robot_arm is not None
        and np.isfinite(np.asarray(robot_arm.center_xyz)).all()
    )
    arm_internal = (
        np.asarray(robot_arm.center_xyz, dtype=np.float32)
        if arm_valid and robot_arm is not None
        else None
    )
    arm_wire = (
        _coordinate_xyz(arm_internal, coordinate_mode)
        if arm_internal is not None
        else None
    )

    obstacle_payloads: list[dict[str, Any]] = []
    for track in tracks:
        center_internal = np.asarray(track.position_xyz, dtype=np.float32).reshape(3)
        velocity_internal = np.asarray(track.velocity_xyz, dtype=np.float32).reshape(3)
        if not (
            np.isfinite(center_internal).all()
            and np.isfinite(velocity_internal).all()
        ):
            continue
        center_wire = _coordinate_xyz(center_internal, coordinate_mode)
        velocity_wire = _coordinate_xyz(velocity_internal, coordinate_mode)
        relation: dict[str, Any] = {
            "track_id": int(track.track_id),
            "class_name": str(track.class_name),
            "obstacle_center_m": _xyz_object(center_wire),
            "velocity_mps": _xyz_object(velocity_wire),
            "radius_m": float(track.radius),
            "confidence": float(track.confidence),
            "hit_count": int(track.hit_count),
            "missing_count": int(track.missing_count),
            "fresh_measurement": bool(track.missing_count == 0),
            "motion_state": str(track.motion_state),
        }
        if arm_internal is not None and arm_wire is not None:
            delta_internal = center_internal - arm_internal
            delta_wire = _coordinate_xyz(delta_internal, coordinate_mode)
            center_distance = float(np.linalg.norm(delta_internal))
            planar_distance = float(np.linalg.norm(delta_internal[:2]))
            relation.update(
                {
                    "delta_from_arm_m": _xyz_object(delta_wire),
                    "center_distance_m": center_distance,
                    "planar_distance_m": planar_distance,
                    "surface_clearance_m": max(
                        0.0,
                        center_distance - max(float(track.radius), 0.0),
                    ),
                }
            )
        else:
            relation.update(
                {
                    "delta_from_arm_m": None,
                    "center_distance_m": None,
                    "planar_distance_m": None,
                    "surface_clearance_m": None,
                }
            )
        obstacle_payloads.append(relation)

    obstacle_payloads.sort(
        key=lambda item: (
            float("inf")
            if item["center_distance_m"] is None
            else float(item["center_distance_m"]),
            int(item["track_id"]),
        )
    )
    nearest = (
        {
            "track_id": obstacle_payloads[0]["track_id"],
            "class_name": obstacle_payloads[0]["class_name"],
            "center_distance_m": obstacle_payloads[0]["center_distance_m"],
            "surface_clearance_m": obstacle_payloads[0]["surface_clearance_m"],
        }
        if obstacle_payloads and arm_valid
        else None
    )
    status = (
        "perception_stale"
        if perception_age_sec > 0.5
        else "arm_not_localized"
        if not arm_valid
        else "no_obstacles"
        if not obstacle_payloads
        else "tracking"
    )
    return {
        "schema": "realtime_safety/arm_obstacle_relationships",
        "schema_version": 1,
        "sequence": int(sequence),
        "perception_sequence": int(perception_sequence),
        "perception_age_sec": max(0.0, float(perception_age_sec)),
        "frame_id": frame_id,
        "coordinate_mode": coordinate_mode,
        "coordinate_convention": (
            "x_right_y_forward_z_down_m"
            if coordinate_mode == "camera_y_forward"
            else "x_right_y_forward_z_up_m"
        ),
        "published_at_unix_sec": time.time(),
        "source_timestamp_sec": (
            None if source_timestamp is None else float(source_timestamp)
        ),
        "status": status,
        "arm_valid": arm_valid,
        "arm": (
            {
                "center_m": _xyz_object(arm_wire),
                "confidence": float(robot_arm.confidence),
                "held_frames": int(robot_arm.held_frames),
                "fresh_measurement": bool(robot_arm.held_frames == 0),
            }
            if arm_valid and robot_arm is not None and arm_wire is not None
            else None
        ),
        "obstacle_count": len(obstacle_payloads),
        "nearest_obstacle": nearest,
        "obstacles": obstacle_payloads,
    }


class ArmObstacleRelationshipPublisher:
    """Publish arm/obstacle centers and distances as versioned JSON."""

    def __init__(
        self,
        topic: str = "/realtime_safety/arm_obstacle_relationships",
        frame_id: str = "realtime_safety_frame",
        node_name: str = "realtime_safety_arm_obstacle_relationship_publisher",
        max_rate_hz: float | None = 12.0,
        coordinate_mode: str = "camera_y_forward",
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError(
                "Relationship topic must be an absolute ROS name without whitespace"
            )
        if max_rate_hz is not None and max_rate_hz <= 0:
            raise ValueError("Relationship publication rate must be positive")
        if coordinate_mode not in {"internal_z_up", "camera_y_forward"}:
            raise ValueError(
                f"Unsupported relationship coordinate mode: {coordinate_mode}"
            )
        self.topic = topic
        self.frame_id = frame_id
        self.node_name = node_name
        self.max_rate_hz = max_rate_hz
        self.coordinate_mode = coordinate_mode
        self._minimum_interval = (
            0.0 if max_rate_hz is None else 1.0 / max_rate_hz
        )
        self._last_publish_time = 0.0
        self._diagnostic_window_start = 0.0
        self._diagnostic_publish_count = 0
        self._sequence = 0
        self._perception_sequence = 0
        self.publish_rate_hz = 0.0
        self.matched_subscriptions = 0
        self.last_obstacle_count = 0
        self.last_status = "not_started"
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publisher: Any | None = None
        self._string_type: Any | None = None
        self._condition = threading.Condition()
        self._latest_state: (
            tuple[RobotArmState | None, list[Track3DState], float | None, int, float]
            | None
        ) = None
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import String

        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node(self.node_name, context=runtime.context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        publisher = node.create_publisher(String, self.topic, qos)
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._publisher = publisher
        self._string_type = String
        self._stop_event.clear()
        if self.max_rate_hz is not None:
            self._worker_thread = threading.Thread(
                target=self._publish_worker,
                name=f"{self.node_name}-worker",
                daemon=True,
            )
            self._worker_thread.start()
        LOGGER.info(
            "Publishing ROS 2 arm-obstacle relationships on %s "
            "(frame=%s, coordinates=%s, domain=%s, localhost_only=%s)",
            self.topic,
            self.frame_id,
            self.coordinate_mode,
            os.environ.get("ROS_DOMAIN_ID", "0"),
            os.environ.get("ROS_LOCALHOST_ONLY", "0"),
        )

    def publish(
        self,
        robot_arm: RobotArmState | None,
        tracks: list[Track3DState],
        *,
        source_timestamp: float | None = None,
    ) -> bool:
        if self._node is None or self._publisher is None:
            raise RuntimeError("Relationship publisher has not been started")
        updated_at = time.perf_counter()
        self._perception_sequence += 1
        state = (
            robot_arm,
            list(tracks),
            source_timestamp,
            self._perception_sequence,
            updated_at,
        )
        # Unit tests and embedded users that inject handles without start()
        # retain immediate behavior. The live publisher uses a fixed-rate
        # worker so control liveness is independent of YOLO/GPU cadence.
        if self._worker_thread is None:
            if (
                self._minimum_interval > 0.0
                and updated_at - self._last_publish_time < self._minimum_interval
            ):
                return False
            self._publish_now(state)
            return True
        with self._condition:
            self._latest_state = state
            self._condition.notify_all()
        return True

    def _publish_worker(self) -> None:
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            wait = max(0.0, next_deadline - time.perf_counter())
            if self._stop_event.wait(wait):
                return
            with self._condition:
                state = self._latest_state
            if state is not None:
                try:
                    self._publish_now(state)
                except Exception:
                    LOGGER.exception(
                        "Arm-obstacle relationship publication failed on %s",
                        self.topic,
                    )
            next_deadline = max(
                next_deadline + max(self._minimum_interval, 0.001),
                time.perf_counter(),
            )

    def _publish_now(
        self,
        state: tuple[
            RobotArmState | None,
            list[Track3DState],
            float | None,
            int,
            float,
        ],
    ) -> None:
        robot_arm, tracks, source_timestamp, perception_sequence, updated_at = state
        now = time.perf_counter()
        self._sequence += 1
        payload = build_relationship_payload(
            robot_arm,
            tracks,
            frame_id=self.frame_id,
            coordinate_mode=self.coordinate_mode,
            source_timestamp=source_timestamp,
            sequence=self._sequence,
            perception_sequence=perception_sequence,
            perception_age_sec=now - updated_at,
        )
        message = self._string_type()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._publisher.publish(message)
        self._last_publish_time = now
        self.last_obstacle_count = int(payload["obstacle_count"])
        self.last_status = str(payload["status"])
        self._diagnostic_publish_count += 1
        if self._diagnostic_window_start <= 0.0:
            self._diagnostic_window_start = now
        elapsed = now - self._diagnostic_window_start
        if elapsed >= 2.0:
            self.publish_rate_hz = self._diagnostic_publish_count / elapsed
            get_subscription_count = getattr(
                self._publisher,
                "get_subscription_count",
                None,
            )
            self.matched_subscriptions = (
                int(get_subscription_count())
                if callable(get_subscription_count)
                else 0
            )
            LOGGER.info(
                "Relationship diagnostics topic=%s status=%s obstacles=%d "
                "rate=%.2fHz matched_subscriptions=%d",
                self.topic,
                self.last_status,
                self.last_obstacle_count,
                self.publish_rate_hz,
                self.matched_subscriptions,
            )
            self._diagnostic_window_start = now
            self._diagnostic_publish_count = 0
    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._latest_state = None
        runtime = self._runtime
        if runtime is not None and self._node is not None:
            runtime.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)
        self._runtime = None
        self._node = None
        self._publisher = None
        self._string_type = None
        self._last_publish_time = 0.0
