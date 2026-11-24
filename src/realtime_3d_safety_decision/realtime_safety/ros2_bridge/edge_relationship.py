from __future__ import annotations

"""Adapt EdgeTAM's native 3D tracks to the legacy relationship contract.

The Edge tracker already owns the timestamp-aware 3D Kalman filter.  This
module deliberately does not track the obstacle a second time: it converts
``TrackedObstacleArray`` samples into the ``Track3DState`` representation used
by :mod:`relationship_publisher`, and arbitrates which backend may update that
publisher.

ROS imports are kept inside :meth:`EdgeTAMRelationshipBridge.start` so the
coordinate conversion and switching rules remain unit-testable without a ROS
installation or a generated message package on ``PYTHONPATH``.
"""

from dataclasses import dataclass
import logging
import threading
from typing import Any, Callable, Iterable

import numpy as np

from realtime_safety.types import BBox3D, RobotArmState, Track3DState


LOGGER = logging.getLogger(__name__)

_ACCEPTED_TRACK_STATES = frozenset({"CONFIRMED", "OCCLUDED", "LOST"})


def _xyz(value: Any) -> np.ndarray:
    result = np.asarray(
        (float(value.x), float(value.y), float(value.z)),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("Edge obstacle geometry must contain finite XYZ values")
    return result


def _to_internal_z_up(
    value: np.ndarray,
    source_coordinate_mode: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(3).copy()
    if source_coordinate_mode == "camera_y_forward":
        # The Koch PointCloud2 wire convention is x-right/y-forward/z-down,
        # while Track3DState and RobotArmState use z-up internally.
        result[2] *= -1.0
    elif source_coordinate_mode == "ros_optical":
        # REP-103 optical x-right/y-down/z-forward -> internal
        # x-right/y-forward/z-up.
        result = result[[0, 2, 1]]
        result[2] *= -1.0
    elif source_coordinate_mode != "internal_z_up":
        raise ValueError(
            "source_coordinate_mode must be internal_z_up, "
            "camera_y_forward, or ros_optical"
        )
    return result


def _stamp_seconds(stamp: Any) -> float:
    seconds = float(getattr(stamp, "sec", 0))
    nanoseconds = float(getattr(stamp, "nanosec", 0))
    result = seconds + nanoseconds * 1e-9
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("Edge obstacle timestamp must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class EdgeRelationshipFrame:
    """One converted Edge obstacle publication."""

    tracks: tuple[Track3DState, ...]
    timestamp: float
    frame_id: str


class EdgeMotionClassifier:
    """Apply the YOLO path's static/dynamic hysteresis to Edge velocities."""

    def __init__(
        self,
        *,
        dynamic_enter_speed: float = 0.15,
        dynamic_exit_speed: float = 0.08,
        minimum_dynamic_hits: int = 3,
    ) -> None:
        if dynamic_enter_speed <= dynamic_exit_speed or dynamic_exit_speed < 0.0:
            raise ValueError(
                "dynamic_enter_speed must be greater than dynamic_exit_speed"
            )
        if minimum_dynamic_hits < 1:
            raise ValueError("minimum_dynamic_hits must be at least one")
        self.dynamic_enter_speed = float(dynamic_enter_speed)
        self.dynamic_exit_speed = float(dynamic_exit_speed)
        self.minimum_dynamic_hits = int(minimum_dynamic_hits)
        self._states: dict[int, tuple[str, int]] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, track_id: int, speed: float) -> str:
        state, dynamic_hits = self._states.get(int(track_id), ("static", 0))
        speed = float(speed)
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError("Edge obstacle speed must be finite and non-negative")
        if state == "static":
            dynamic_hits = dynamic_hits + 1 if speed >= self.dynamic_enter_speed else 0
            if dynamic_hits >= self.minimum_dynamic_hits:
                state = "dynamic"
        elif speed <= self.dynamic_exit_speed:
            dynamic_hits = max(dynamic_hits - 1, 0)
            if dynamic_hits == 0:
                state = "static"
        else:
            dynamic_hits = self.minimum_dynamic_hits
        self._states[int(track_id)] = (state, dynamic_hits)
        return state

    def retain(self, live_track_ids: Iterable[int]) -> None:
        live = {int(track_id) for track_id in live_track_ids}
        self._states = {
            track_id: state
            for track_id, state in self._states.items()
            if track_id in live
        }


def edge_obstacle_array_to_track_states(
    message: Any,
    *,
    source_coordinate_mode: str = "camera_y_forward",
    class_name: str = "hand_candidate",
    accepted_states: Iterable[str] = _ACCEPTED_TRACK_STATES,
    motion_classifier: EdgeMotionClassifier | None = None,
) -> EdgeRelationshipFrame:
    """Convert native Edge tracks without altering their IDs or kinematics.

    ``class_name`` is intentionally configurable.  EdgeTAM is promptable but
    not semantic; callers must only use ``"hand"`` after an upstream hand
    classifier/prompt gate has accepted the track.  Until then the truthful
    default is ``"hand_candidate"``.
    """

    if not class_name.strip():
        raise ValueError("class_name must not be empty")
    if source_coordinate_mode not in {
        "internal_z_up",
        "camera_y_forward",
        "ros_optical",
    }:
        raise ValueError(
            "source_coordinate_mode must be internal_z_up, "
            "camera_y_forward, or ros_optical"
        )
    accepted = {str(state).upper() for state in accepted_states}
    header = message.header
    timestamp = _stamp_seconds(header.stamp)
    frame_id = str(header.frame_id)
    tracks: list[Track3DState] = []
    live_ids: set[int] = set()

    for obstacle in message.obstacles:
        tracking_state = str(obstacle.tracking_state).upper()
        if tracking_state not in accepted:
            continue
        if not bool(getattr(obstacle, "semantic_confirmed", True)):
            continue
        track_id = int(obstacle.track_id)
        live_ids.add(track_id)
        center = _to_internal_z_up(
            _xyz(obstacle.filtered_centroid), source_coordinate_mode
        )
        velocity = _to_internal_z_up(
            _xyz(obstacle.velocity), source_coordinate_mode
        )
        first_corner = _to_internal_z_up(
            _xyz(obstacle.aabb_min), source_coordinate_mode
        )
        second_corner = _to_internal_z_up(
            _xyz(obstacle.aabb_max), source_coordinate_mode
        )
        minimum = np.minimum(first_corner, second_corner)
        maximum = np.maximum(first_corner, second_corner)
        planar_half_size = (maximum[:2] - minimum[:2]) * 0.5
        radius = max(float(np.linalg.norm(planar_half_size)), 0.05)
        uncertainty = max(float(obstacle.uncertainty_margin), 1e-3)
        covariance = np.diag(
            [uncertainty**2] * 3 + [max(uncertainty, 0.05) ** 2] * 3
        ).astype(np.float64)
        speed = float(np.linalg.norm(velocity))
        motion_state = (
            motion_classifier.update(track_id, speed)
            if motion_classifier is not None
            else ("dynamic" if speed >= 0.15 else "static")
        )
        measurement_stamp = _stamp_seconds(obstacle.last_measurement_stamp)
        semantic_class = str(
            getattr(obstacle, "semantic_class", "")
        ).strip()
        tracks.append(
            Track3DState(
                track_id=track_id,
                class_name=semantic_class or class_name,
                position_xyz=center,
                velocity_xyz=velocity,
                acceleration_xyz=np.zeros(3, dtype=np.float32),
                covariance=covariance,
                bbox3d=BBox3D(minimum=minimum, maximum=maximum),
                radius=radius,
                hit_count=max(int(obstacle.hit_count), 0),
                missing_count=max(int(obstacle.missed_frame_count), 0),
                last_timestamp=measurement_stamp,
                motion_state=motion_state,
                confidence=float(np.clip(obstacle.confidence, 0.0, 1.0)),
            )
        )

    if motion_classifier is not None:
        motion_classifier.retain(live_ids)
    tracks.sort(key=lambda track: track.track_id)
    return EdgeRelationshipFrame(tuple(tracks), timestamp, frame_id)


class EdgeTAMRelationshipBridge:
    """Subscribe to Edge tracks and feed the existing relationship publisher.

    The bridge is hot while the application is running, but it only forwards
    a sample when ``active_mode`` is ``"edgetam"`` or ``"pointcloud"``.  A
    switch requires one post-switch Edge sample, so a retained DDS sample from
    the old mode cannot overwrite a newer YOLO relationship.
    """

    def __init__(
        self,
        relationship_publisher: Any,
        *,
        topic: str = "/edgetam_tracker/obstacles",
        source_coordinate_mode: str = "camera_y_forward",
        class_name: str = "hand_candidate",
        initial_mode: str = "edgetam",
        dynamic_enter_speed: float = 0.15,
        dynamic_exit_speed: float = 0.08,
        minimum_dynamic_hits: int = 3,
        maximum_arm_age_sec: float = 0.35,
        manage_publisher_lifecycle: bool = False,
        on_tracks: Callable[[list[Track3DState]], None] | None = None,
        on_frame: Callable[[EdgeRelationshipFrame], None] | None = None,
        robot_arm_provider: Callable[[float], RobotArmState | None] | None = None,
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("Edge obstacle topic must be an absolute ROS name")
        if initial_mode not in {"edgetam", "pointcloud", "yolo"}:
            raise ValueError(f"Unsupported obstacle backend: {initial_mode}")
        if maximum_arm_age_sec < 0.0:
            raise ValueError("maximum_arm_age_sec cannot be negative")
        self.relationship_publisher = relationship_publisher
        self.topic = topic
        self.source_coordinate_mode = source_coordinate_mode
        self.class_name = class_name
        self.maximum_arm_age_sec = float(maximum_arm_age_sec)
        self.manage_publisher_lifecycle = bool(manage_publisher_lifecycle)
        self.on_tracks = on_tracks
        self.on_frame = on_frame
        self.robot_arm_provider = robot_arm_provider
        self._motion_classifier = EdgeMotionClassifier(
            dynamic_enter_speed=dynamic_enter_speed,
            dynamic_exit_speed=dynamic_exit_speed,
            minimum_dynamic_hits=minimum_dynamic_hits,
        )
        self._lock = threading.Lock()
        self._active_mode = initial_mode
        self._robot_arm: RobotArmState | None = None
        self._latest_frame: EdgeRelationshipFrame | None = None
        self._received_sequence = 0
        self._minimum_forward_sequence = 1
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscription: Any | None = None

    @property
    def active_mode(self) -> str:
        with self._lock:
            return self._active_mode

    def set_mode(self, mode: str) -> None:
        if mode not in {"edgetam", "pointcloud", "yolo"}:
            raise ValueError(f"Unsupported obstacle backend: {mode}")
        with self._lock:
            if mode == self._active_mode:
                return
            self._active_mode = mode
            self._minimum_forward_sequence = self._received_sequence + 1
            # A center measured by the backend being left must never be paired
            # with a newly selected source.  The scheduler supplies a fresh
            # arm estimate from the next aligned RGB/point-cloud frame.
            self._robot_arm = None
            self._motion_classifier.reset()

    def update_robot_arm(self, robot_arm: RobotArmState | None) -> None:
        with self._lock:
            self._robot_arm = robot_arm

    @property
    def latest_frame(self) -> EdgeRelationshipFrame | None:
        with self._lock:
            return self._latest_frame

    def handle_message(self, message: Any) -> bool:
        """Convert and forward one sample; public for deterministic tests."""

        with self._lock:
            self._received_sequence += 1
            sequence = self._received_sequence
            mode = self._active_mode
            minimum_sequence = self._minimum_forward_sequence
            robot_arm = self._robot_arm
            if mode not in {"edgetam", "pointcloud"} or sequence < minimum_sequence:
                return False
            frame = edge_obstacle_array_to_track_states(
                message,
                source_coordinate_mode=self.source_coordinate_mode,
                class_name=self.class_name,
                motion_classifier=self._motion_classifier,
            )
            if self.robot_arm_provider is not None:
                provided_arm = self.robot_arm_provider(frame.timestamp)
                if provided_arm is not None:
                    robot_arm = provided_arm
            self._latest_frame = frame
            if (
                robot_arm is not None
                and abs(float(robot_arm.timestamp) - frame.timestamp)
                > self.maximum_arm_age_sec
            ):
                # A stale center is worse than an explicit arm_not_localized
                # state for an avoidance controller.
                robot_arm = None
            # Keep source selection and the underlying latest-state update in
            # one critical section.  Otherwise an in-flight YOLO inference can
            # overwrite the first Edge sample immediately after a GUI switch.
            self.relationship_publisher.publish(
                robot_arm,
                list(frame.tracks),
                source_timestamp=frame.timestamp,
            )
        if self.on_tracks is not None:
            self.on_tracks(list(frame.tracks))
        if self.on_frame is not None:
            self.on_frame(frame)
        return True

    def publish(
        self,
        robot_arm: RobotArmState | None,
        tracks: list[Track3DState],
        *,
        source_timestamp: float | None = None,
    ) -> bool:
        """Drop-in YOLO-side ``ArmObstacleRelationshipPublisher`` method.

        Passing this bridge to ``RealtimePipeline`` in place of the raw
        publisher centralizes arbitration.  Existing scheduler calls remain
        source-compatible and are ignored while Edge owns the relationship
        output.
        """

        with self._lock:
            if self._active_mode != "yolo":
                return False
            return bool(
                self.relationship_publisher.publish(
                    robot_arm,
                    tracks,
                    source_timestamp=source_timestamp,
                )
            )

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

        from realtime_3d_safety_decision.msg import TrackedObstacleArray
        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        if self.manage_publisher_lifecycle:
            self.relationship_publisher.start()
        try:
            runtime = acquire_ros2_runtime()
            node = Node(
                "realtime_safety_edgetam_relationship_bridge",
                context=runtime.context,
            )
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            subscription = node.create_subscription(
                TrackedObstacleArray,
                self.topic,
                self._on_message,
                qos,
            )
            runtime.add_node(node)
            self._runtime = runtime
            self._node = node
            self._subscription = subscription
        except Exception:
            if self.manage_publisher_lifecycle:
                self.relationship_publisher.close()
            raise
        LOGGER.info("Subscribing to Edge obstacle relationships on %s", self.topic)

    def _on_message(self, message: Any) -> None:
        try:
            self.handle_message(message)
        except Exception:
            LOGGER.exception("Invalid Edge obstacle relationship sample on %s", self.topic)

    def close(self) -> None:
        runtime = self._runtime
        node = self._node
        if runtime is not None and node is not None:
            runtime.remove_node(node)
        if node is not None:
            node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)
        self._runtime = None
        self._node = None
        self._subscription = None
        if self.manage_publisher_lifecycle:
            self.relationship_publisher.close()
