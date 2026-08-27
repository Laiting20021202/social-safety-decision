from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from realtime_safety.ros2_bridge.stamps import (
    exact_source_timestamp_or_now,
    source_timestamp_or_now,
)
from realtime_safety.types import PointCloudFrame

LOGGER = logging.getLogger(__name__)

_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("rgb", "<u4"),
    ]
)


def _pack_rgb_points(
    points: np.ndarray,
    colors: np.ndarray,
    coordinate_mode: str = "internal_z_up",
) -> tuple[bytes, int]:
    """Pack x/y/z and 0xRRGGBB into a PointCloud2-compatible byte buffer."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if coordinate_mode not in {
        "internal_z_up",
        "camera_y_forward",
        "ros_optical",
    }:
        raise ValueError(f"Unsupported point-cloud coordinate mode: {coordinate_mode}")
    if coordinate_mode == "camera_y_forward" and len(points):
        # Internal reconstruction uses z-up. Koch VAMP's camera_y_forward
        # contract is x-right, y-forward/depth, z-down.
        points = points.copy()
        points[:, 2] *= -1.0
    elif coordinate_mode == "ros_optical" and len(points):
        # Internal reconstruction is x-right/y-forward/z-up. REP-103 optical
        # is x-right/y-down/z-forward.
        points = np.column_stack(
            (points[:, 0], -points[:, 2], points[:, 1])
        ).astype(np.float32, copy=False)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    count = min(len(points), len(colors))
    points, colors = points[:count], colors[:count]
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    packed = np.empty(len(points), dtype=_POINT_DTYPE)
    if len(points):
        packed["x"], packed["y"], packed["z"] = points.T
    color32 = (
        colors[:, 0].astype(np.uint32) << 16
        | colors[:, 1].astype(np.uint32) << 8
        | colors[:, 2].astype(np.uint32)
    )
    packed["rgb"] = color32
    return packed.tobytes(), len(packed)


class PointCloudTopicPublisher:
    """Publish the latest reconstructed cloud as a LAN-discoverable ROS 2 topic."""

    def __init__(
        self,
        topic: str = "/realtime_safety/pointcloud",
        frame_id: str = "realtime_safety_frame",
        node_name: str = "realtime_safety_pointcloud_publisher",
        max_rate_hz: float | None = None,
        publish_empty: bool = False,
        coordinate_mode: str = "internal_z_up",
        preserve_source_timestamp: bool = False,
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("Point-cloud topic must be an absolute ROS name without whitespace")
        self.topic = topic
        self.frame_id = frame_id
        self.node_name = node_name
        if max_rate_hz is not None and max_rate_hz <= 0:
            raise ValueError("Point-cloud publication rate must be positive")
        self.max_rate_hz = max_rate_hz
        self.publish_empty = bool(publish_empty)
        if coordinate_mode not in {
            "internal_z_up",
            "camera_y_forward",
            "ros_optical",
        }:
            raise ValueError(f"Unsupported point-cloud coordinate mode: {coordinate_mode}")
        self.coordinate_mode = coordinate_mode
        self.preserve_source_timestamp = bool(preserve_source_timestamp)
        self._minimum_interval = 0.0 if max_rate_hz is None else 1.0 / max_rate_hz
        self._last_publish_time = 0.0
        self._diagnostic_window_start = 0.0
        self._diagnostic_publish_count = 0
        self.last_width = 0
        self.last_data_bytes = 0
        self.publish_rate_hz = 0.0
        self.matched_subscriptions = 0
        self.last_publish_duration_ms = 0.0
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publisher: Any | None = None
        self._pointcloud_type: Any | None = None
        self._pointfield_type: Any | None = None
        self._condition = threading.Condition()
        self._pending_cloud: PointCloudFrame | None = None
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField
        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node(self.node_name, context=runtime.context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            # A reliable writer is compatible with both RViz's default reliable
            # reader and the Koch computer's best-effort sensor-data reader.
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        publisher = node.create_publisher(PointCloud2, self.topic, qos)
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._publisher = publisher
        self._pointcloud_type = PointCloud2
        self._pointfield_type = PointField
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._publish_worker,
            name=f"{self.node_name}-worker",
            daemon=True,
        )
        self._worker_thread.start()
        LOGGER.info(
            "Publishing ROS 2 PointCloud2 on %s "
            "(frame=%s, coordinates=%s, domain=%s, localhost_only=%s)",
            self.topic,
            self.frame_id,
            self.coordinate_mode,
            os.environ.get("ROS_DOMAIN_ID", "0"),
            os.environ.get("ROS_LOCALHOST_ONLY", "0"),
        )

    def publish(self, cloud: PointCloudFrame) -> None:
        if self._node is None or self._publisher is None:
            raise RuntimeError("Point-cloud publisher has not been started")
        # Unit tests and deliberately embedded publishers may inject the ROS
        # handles without calling start(); retain synchronous behavior there.
        if self._worker_thread is None:
            self._publish_now(cloud)
            return
        with self._condition:
            self._pending_cloud = cloud
            self._condition.notify()

    def _publish_worker(self) -> None:
        """Serialize DDS messages off the depth/YOLO inference threads."""

        while not self._stop_event.is_set():
            with self._condition:
                while self._pending_cloud is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                cloud = self._pending_cloud
                self._pending_cloud = None

            wait = self._minimum_interval - (
                time.perf_counter() - self._last_publish_time
            )
            if wait > 0 and self._stop_event.wait(wait):
                return
            # While rate-limited, keep only the newest frame. This prevents DDS
            # serialization from building an old point-cloud backlog.
            with self._condition:
                if self._pending_cloud is not None:
                    cloud = self._pending_cloud
                    self._pending_cloud = None
            if cloud is None:
                continue
            try:
                self._publish_now(cloud)
            except Exception:
                LOGGER.exception("Point-cloud publication failed on %s", self.topic)

    def _publish_now(self, cloud: PointCloudFrame) -> None:
        now = time.perf_counter()
        PointCloud2 = self._pointcloud_type
        PointField = self._pointfield_type
        data, count = _pack_rgb_points(
            cloud.points,
            cloud.colors,
            coordinate_mode=self.coordinate_mode,
        )
        if count == 0 and not self.publish_empty:
            return
        message = PointCloud2()
        stamp_converter = (
            exact_source_timestamp_or_now
            if self.preserve_source_timestamp
            else source_timestamp_or_now
        )
        message.header.stamp = stamp_converter(self._node, cloud.timestamp)
        message.header.frame_id = self.frame_id
        message.height = 1
        message.width = count
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = _POINT_DTYPE.itemsize
        message.row_step = message.point_step * count
        message.data = data
        message.is_dense = True
        publish_started = time.perf_counter()
        self._publisher.publish(message)
        self.last_publish_duration_ms = (
            time.perf_counter() - publish_started
        ) * 1000.0
        self._last_publish_time = now
        self.last_width = count
        self.last_data_bytes = len(data)
        self._diagnostic_publish_count += 1
        if self._diagnostic_window_start <= 0.0:
            self._diagnostic_window_start = now
        diagnostic_elapsed = now - self._diagnostic_window_start
        if diagnostic_elapsed >= 2.0:
            self.publish_rate_hz = self._diagnostic_publish_count / diagnostic_elapsed
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
            executor_alive = bool(
                self._runtime is not None
                and self._runtime.thread.is_alive()
            )
            LOGGER.info(
                "PointCloud2 diagnostics topic=%s width=%d height=1 "
                "point_step=%d row_step=%d bytes=%d rate=%.2fHz "
                "matched_subscriptions=%d publish_ms=%.3f executor_alive=%s "
                "coordinates=%s",
                self.topic,
                count,
                message.point_step,
                message.row_step,
                len(data),
                self.publish_rate_hz,
                self.matched_subscriptions,
                self.last_publish_duration_ms,
                executor_alive,
                self.coordinate_mode,
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
        self._pending_cloud = None
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
        self._pointcloud_type = None
        self._pointfield_type = None
        self._last_publish_time = 0.0
        self._diagnostic_window_start = 0.0
        self._diagnostic_publish_count = 0
        self.last_width = 0
        self.last_data_bytes = 0
        self.publish_rate_hz = 0.0
        self.matched_subscriptions = 0
        self.last_publish_duration_ms = 0.0
