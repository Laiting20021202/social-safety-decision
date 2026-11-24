from __future__ import annotations

"""Runtime mux for one controller-facing obstacle PointCloud2 topic.

EdgeTAM/3D and the retained YOLO/3D system publish private candidate topics.
This bridge is the sole writer to the historical controller topic and changes
source only after the newly requested pipeline produces its first message.
The previous source therefore remains live while a model is loading.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Literal


LOGGER = logging.getLogger(__name__)

ObstaclePipelineMode = Literal["edgetam", "pointcloud", "yolo"]
_VALID_MODES = frozenset({"edgetam", "pointcloud", "yolo"})


def _source_for_mode(mode: ObstaclePipelineMode) -> str:
    return "yolo" if mode == "yolo" else "edge"


@dataclass(frozen=True, slots=True)
class ObstacleCloudMuxStatus:
    requested_mode: ObstaclePipelineMode
    active_mode: ObstaclePipelineMode
    state: Literal["waiting", "active", "error"]
    message: str


class ObstacleCloudMux:
    """Select Edge/point-cloud or YOLO candidate clouds without topic races."""

    valid_modes = ("edgetam", "pointcloud", "yolo")

    def __init__(
        self,
        *,
        edge_topic: str = "/edgetam_tracker/obstacle_cloud",
        yolo_topic: str = "/realtime_safety/yolo_obstacles/candidate_cloud",
        output_topic: str = "/realtime_safety/yolo_obstacles/pointcloud",
        initial_mode: ObstaclePipelineMode = "edgetam",
        on_status: Callable[[ObstacleCloudMuxStatus], None] | None = None,
        node_name: str = "realtime_safety_obstacle_cloud_mux",
        stale_timeout_sec: float = 1.0,
    ) -> None:
        for label, value in (
            ("Edge source topic", edge_topic),
            ("YOLO source topic", yolo_topic),
            ("Output topic", output_topic),
        ):
            if not value.startswith("/") or any(char.isspace() for char in value):
                raise ValueError(f"{label} must be an absolute ROS name")
        if len({edge_topic, yolo_topic, output_topic}) != 3:
            raise ValueError("Obstacle mux source and output topics must be distinct")
        if not node_name or any(char.isspace() for char in node_name):
            raise ValueError("ROS node name must not be empty or contain whitespace")
        if stale_timeout_sec <= 0.0:
            raise ValueError("Obstacle mux stale_timeout_sec must be positive")
        initial = self._validate_mode(initial_mode)
        self.edge_topic = edge_topic
        self.yolo_topic = yolo_topic
        self.output_topic = output_topic
        self.node_name = node_name
        self.stale_timeout_sec = float(stale_timeout_sec)
        self._on_status = on_status
        self._lock = threading.RLock()
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publisher: Any | None = None
        self._edge_subscription: Any | None = None
        self._yolo_subscription: Any | None = None
        self._watchdog_timer: Any | None = None
        self._last_received_at: dict[str, float | None] = {
            "edge": None,
            "yolo": None,
        }
        self._last_message: dict[str, Any | None] = {
            "edge": None,
            "yolo": None,
        }
        self._requested_mode = initial
        self._active_mode = initial
        self._active_source = _source_for_mode(initial)
        self._pending_source: str | None = None
        self._state: Literal["waiting", "active", "error"] = "waiting"
        self._closed = False

    @staticmethod
    def _validate_mode(mode: str) -> ObstaclePipelineMode:
        normalized = str(mode).strip().lower()
        if normalized not in _VALID_MODES:
            raise ValueError(
                "Obstacle pipeline mode must be edgetam, pointcloud, or yolo"
            )
        return normalized  # type: ignore[return-value]

    @property
    def requested_mode(self) -> ObstaclePipelineMode:
        with self._lock:
            return self._requested_mode

    @property
    def active_mode(self) -> ObstaclePipelineMode:
        with self._lock:
            return self._active_mode

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._node is not None

    def start(self) -> None:
        with self._lock:
            if self._node is not None:
                return

        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import PointCloud2

        from realtime_safety.ros2_bridge.runtime import (
            acquire_ros2_runtime,
            release_ros2_runtime,
        )

        runtime = acquire_ros2_runtime()
        node: Any | None = None
        try:
            node = Node(self.node_name, context=runtime.context)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            publisher = node.create_publisher(PointCloud2, self.output_topic, qos)
            edge_subscription = node.create_subscription(
                PointCloud2,
                self.edge_topic,
                lambda message: self._on_source(message, "edge"),
                qos,
            )
            yolo_subscription = node.create_subscription(
                PointCloud2,
                self.yolo_topic,
                lambda message: self._on_source(message, "yolo"),
                qos,
            )
            watchdog_timer = node.create_timer(0.2, self._check_stale)
            runtime.add_node(node)
        except Exception:
            if node is not None:
                node.destroy_node()
            release_ros2_runtime(runtime)
            raise

        with self._lock:
            self._runtime = runtime
            self._node = node
            self._publisher = publisher
            self._edge_subscription = edge_subscription
            self._yolo_subscription = yolo_subscription
            self._watchdog_timer = watchdog_timer
            self._closed = False
            status = self._status_locked(
                "Waiting for the first selected obstacle cloud"
            )
        self._notify(status)

    def request_mode(self, mode: str) -> bool:
        requested = self._validate_mode(mode)
        with self._lock:
            if self._closed:
                raise RuntimeError("Obstacle cloud mux is closed")
            self._requested_mode = requested
            requested_source = _source_for_mode(requested)
            if requested_source == self._active_source:
                self._active_mode = requested
                self._pending_source = None
                self._state = "active"
                status = self._status_locked(
                    f"{requested} uses the active obstacle-cloud source"
                )
            else:
                self._pending_source = requested_source
                self._state = "waiting"
                status = self._status_locked(
                    f"Waiting for the first {requested} obstacle cloud; "
                    f"{self._active_mode} output remains active"
                )
        self._notify(status)
        return True

    def _on_source(self, message: Any, source: str) -> None:
        status: ObstacleCloudMuxStatus | None = None
        with self._lock:
            self._last_received_at[source] = time.monotonic()
            self._last_message[source] = message
            publisher = self._publisher
            if self._closed or publisher is None:
                return
            if source == self._pending_source:
                self._active_source = source
                self._active_mode = self._requested_mode
                self._pending_source = None
                self._state = "active"
                status = self._status_locked(
                    f"{self._active_mode} obstacle cloud is active"
                )
            if source != self._active_source:
                return
            if self._state == "error":
                if self._pending_source is None:
                    self._state = "active"
                    status = self._status_locked(
                        f"{self._active_mode} obstacle cloud resumed"
                    )
                else:
                    self._state = "waiting"
                    status = self._status_locked(
                        f"{self._active_mode} obstacle cloud resumed; "
                        f"waiting for {self._requested_mode}"
                    )
            try:
                publisher.publish(message)
            except Exception as exc:
                self._state = "error"
                status = self._status_locked(
                    f"Could not publish selected obstacle cloud: {exc}"
                )
        if status is not None:
            self._notify(status)

    def _check_stale(self, now: float | None = None) -> None:
        """Report stale selected output without changing the selected model.

        A downstream controller already treats stale perception as STOP.
        Automatically falling back and then returning to YOLO made the two
        models visibly oscillate. Only :meth:`request_mode` may switch sources.
        """

        current = time.monotonic() if now is None else float(now)
        status: ObstacleCloudMuxStatus | None = None
        with self._lock:
            if self._closed or self._active_source != "yolo":
                return
            yolo_at = self._last_received_at["yolo"]
            if yolo_at is None or current - yolo_at <= self.stale_timeout_sec:
                return
            self._state = "error"
            status = self._status_locked(
                "Selected YOLO obstacle cloud is stale; output is held until "
                "YOLO resumes or the operator selects another model"
            )
        if status is not None:
            self._notify(status)

    def _status_locked(self, message: str) -> ObstacleCloudMuxStatus:
        return ObstacleCloudMuxStatus(
            requested_mode=self._requested_mode,
            active_mode=self._active_mode,
            state=self._state,
            message=message,
        )

    def _notify(self, status: ObstacleCloudMuxStatus) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(status)
        except Exception:
            LOGGER.exception("Obstacle cloud mux status callback failed")

    def close(self) -> None:
        with self._lock:
            if self._closed and self._node is None:
                return
            self._closed = True
            runtime = self._runtime
            node = self._node
            publisher = self._publisher
            edge_subscription = self._edge_subscription
            yolo_subscription = self._yolo_subscription
            watchdog_timer = self._watchdog_timer
            self._runtime = None
            self._node = None
            self._publisher = None
            self._edge_subscription = None
            self._yolo_subscription = None
            self._watchdog_timer = None

        if runtime is not None and node is not None:
            runtime.remove_node(node)
        if node is not None and edge_subscription is not None:
            node.destroy_subscription(edge_subscription)
        if node is not None and yolo_subscription is not None:
            node.destroy_subscription(yolo_subscription)
        if node is not None and watchdog_timer is not None:
            node.destroy_timer(watchdog_timer)
        if node is not None and publisher is not None:
            node.destroy_publisher(publisher)
        if node is not None:
            node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)


__all__ = [
    "ObstacleCloudMux",
    "ObstacleCloudMuxStatus",
    "ObstaclePipelineMode",
]
