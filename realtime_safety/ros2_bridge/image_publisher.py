from __future__ import annotations

import time
from typing import Any

from realtime_safety.types import FramePacket


class ImageTopicPublisher:
    """Publish the latest decoded camera frame as a local ROS 2 Image topic."""

    def __init__(
        self,
        topic: str = "/realtime_safety/camera/image_raw",
        frame_id: str = "koch_webcam_optical_frame",
        node_name: str = "realtime_safety_camera_preview_publisher",
        max_rate_hz: float = 10.0,
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("Camera preview topic must be an absolute ROS name without whitespace")
        if max_rate_hz <= 0:
            raise ValueError("Camera preview publication rate must be positive")
        self.topic = topic
        self.frame_id = frame_id
        self.node_name = node_name
        self._minimum_interval = 1.0 / float(max_rate_hz)
        self._last_publish_time = 0.0
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publisher: Any | None = None
        self._image_type: Any | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node(self.node_name, context=runtime.context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        publisher = node.create_publisher(Image, self.topic, qos)
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._publisher = publisher
        self._image_type = Image

    def publish(self, frame: FramePacket) -> None:
        if self._node is None or self._publisher is None or self._image_type is None:
            raise RuntimeError("Camera preview publisher has not been started")
        now = time.perf_counter()
        if now - self._last_publish_time < self._minimum_interval:
            return
        height, width = frame.bgr.shape[:2]
        message = self._image_type()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.height = height
        message.width = width
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = width * 3
        message.data = frame.bgr.tobytes()
        self._publisher.publish(message)
        self._last_publish_time = now

    def close(self) -> None:
        runtime = self._runtime
        if runtime is not None and self._node is not None:
            runtime.remove_node(self._node)
        if self._node is not None and self._publisher is not None:
            self._node.destroy_publisher(self._publisher)
        if self._node is not None:
            self._node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)
        self._runtime = None
        self._node = None
        self._publisher = None
        self._image_type = None
        self._last_publish_time = 0.0
