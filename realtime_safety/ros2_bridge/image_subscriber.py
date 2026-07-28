from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def image_message_to_bgr(message: Any) -> np.ndarray:
    """Convert the common sensor_msgs/Image encodings into contiguous BGR."""

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid ROS image dimensions: {width}x{height}, step={step}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(f"Truncated ROS image: received {raw.size} bytes, expected {required}")
    rows = raw[:required].reshape(height, step)

    if encoding == "rgb8":
        rgb = rows[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding in {"bgr8", "8uc3"}:
        return np.ascontiguousarray(rows[:, : width * 3].reshape(height, width, 3))
    if encoding == "rgba8":
        rgba = rows[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        bgra = rows[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if encoding in {"mono8", "8uc1"}:
        mono = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if encoding in {"yuv422", "yuv422_yuy2", "yuyv", "yuy2"}:
        yuyv = rows[:, : width * 2].reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")


def compressed_image_message_to_bgr(message: Any) -> np.ndarray:
    """Decode a sensor_msgs/CompressedImage JPEG/PNG into contiguous BGR."""

    encoded = np.frombuffer(message.data, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("Empty ROS compressed image")
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"Cannot decode ROS compressed image format: {message.format!r}")
    return np.ascontiguousarray(bgr)


class Ros2ImageCapture:
    """Latest-only ROS 2 Image subscriber with its own rclpy context."""

    def __init__(
        self,
        topic: str,
        node_name: str = "realtime_safety_camera_subscriber",
        stale_after_seconds: float = 2.5,
        preview_topic: str | None = None,
        preview_rate_hz: float = 10.0,
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("ROS image topic must be an absolute name without whitespace")
        if preview_topic is not None and (
            not preview_topic.startswith("/") or any(char.isspace() for char in preview_topic)
        ):
            raise ValueError("ROS preview topic must be an absolute name without whitespace")
        self.topic = topic
        self.node_name = node_name
        self.is_compressed = topic.rstrip("/").endswith("/compressed")
        self.stale_after_seconds = max(float(stale_after_seconds), 0.1)
        self.preview_topic = preview_topic
        self._preview_minimum_interval = 1.0 / max(float(preview_rate_hz), 0.1)
        self._lock = threading.Lock()
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscription: Any | None = None
        self._preview_publisher: Any | None = None
        self._image_type: Any | None = None
        self._last_preview_publish = 0.0
        self._latest: tuple[np.ndarray, float, float] | None = None
        self._version = 0
        self._read_version = 0
        self._last_received_at = 0.0
        self._previous_received_at = 0.0
        self._estimated_fps = 0.0
        self._last_decode_error = ""

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image
        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node(self.node_name, context=runtime.context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            # The Koch camera publishes RELIABLE.  A raw 320x240 RGB frame is
            # split across many UDP datagrams; BEST_EFFORT can therefore lose
            # the complete frame when only one Wi-Fi fragment is dropped.
            # RELIABLE retransmits missing fragments while depth=1 prevents a
            # slow consumer from accumulating stale video.
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        message_type = CompressedImage if self.is_compressed else Image
        subscription = node.create_subscription(message_type, self.topic, self._on_image, qos)
        preview_publisher = (
            node.create_publisher(Image, self.preview_topic, qos)
            if self.preview_topic is not None
            else None
        )
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._subscription = subscription
        self._preview_publisher = preview_publisher
        self._image_type = Image
        transport = "compressed" if self.is_compressed else "raw"
        LOGGER.info(
            "Subscribing to ROS 2 %s camera topic %s with reliable latest-frame QoS",
            transport,
            self.topic,
        )

    def _on_image(self, message: Any) -> None:
        try:
            bgr = (
                compressed_image_message_to_bgr(message)
                if self.is_compressed
                else image_message_to_bgr(message)
            )
        except (ValueError, cv2.error) as exc:
            detail = str(exc)
            if detail != self._last_decode_error:
                LOGGER.warning("Cannot decode %s: %s", self.topic, detail)
                self._last_decode_error = detail
            return
        now = time.perf_counter()
        header_stamp = (
            float(message.header.stamp.sec) + float(message.header.stamp.nanosec) / 1_000_000_000.0
        )
        with self._lock:
            if self._previous_received_at > 0.0:
                delta = now - self._previous_received_at
                if 0.001 < delta < 2.0:
                    instantaneous = 1.0 / delta
                    self._estimated_fps = (
                        instantaneous
                        if self._estimated_fps <= 0.0
                        else 0.9 * self._estimated_fps + 0.1 * instantaneous
                    )
            self._previous_received_at = now
            self._last_received_at = now
            self._latest = (bgr, header_stamp, self._estimated_fps)
            self._version += 1
            self._last_decode_error = ""
        self._publish_preview(message, bgr, now)

    def _publish_preview(self, source_message: Any, bgr: np.ndarray, now: float) -> None:
        if (
            self._preview_publisher is None
            or self._image_type is None
            or now - self._last_preview_publish < self._preview_minimum_interval
        ):
            return
        preview = self._image_type()
        preview.header = source_message.header
        preview.height, preview.width = bgr.shape[:2]
        preview.encoding = "bgr8"
        preview.is_bigendian = False
        preview.step = int(preview.width * 3)
        preview.data = bgr.tobytes()
        self._preview_publisher.publish(preview)
        self._last_preview_publish = now

    @property
    def is_connected(self) -> bool:
        with self._lock:
            last_received_at = self._last_received_at
        return last_received_at > 0.0 and time.perf_counter() - last_received_at <= self.stale_after_seconds

    def read_latest(self) -> tuple[np.ndarray, float, float] | None:
        with self._lock:
            if self._latest is None or self._version == self._read_version:
                return None
            self._read_version = self._version
            bgr, source_timestamp, fps = self._latest
            return bgr.copy(), source_timestamp, fps

    def close(self) -> None:
        runtime = self._runtime
        if runtime is not None and self._node is not None:
            runtime.remove_node(self._node)
        if self._node is not None and self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
        if self._node is not None and self._preview_publisher is not None:
            self._node.destroy_publisher(self._preview_publisher)
        if self._node is not None:
            self._node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)
        self._runtime = None
        self._node = None
        self._subscription = None
        self._preview_publisher = None
        self._image_type = None
        self._last_preview_publish = 0.0
        with self._lock:
            self._latest = None
            self._last_received_at = 0.0
            self._previous_received_at = 0.0
            self._estimated_fps = 0.0
            self._version = 0
            self._read_version = 0
