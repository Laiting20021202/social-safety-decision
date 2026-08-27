from __future__ import annotations

import math
import time
from typing import Any

from realtime_safety.ros2_bridge.stamps import source_timestamp_or_now
from realtime_safety.types import FramePacket


class ImageTopicPublisher:
    """Publish the latest decoded camera frame as a local ROS 2 Image topic."""

    def __init__(
        self,
        topic: str = "/realtime_safety/camera/image_raw",
        frame_id: str = "koch_webcam_optical_frame",
        node_name: str = "realtime_safety_camera_preview_publisher",
        max_rate_hz: float = 10.0,
        camera_info_topic: str | None = None,
        focal_length_x: float | None = None,
        focal_length_y: float | None = None,
        principal_point_x: float | None = None,
        principal_point_y: float | None = None,
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("Camera preview topic must be an absolute ROS name without whitespace")
        if max_rate_hz <= 0:
            raise ValueError("Camera preview publication rate must be positive")
        normalized_camera_info_topic = (
            None
            if camera_info_topic is None or not str(camera_info_topic).strip()
            else str(camera_info_topic).strip()
        )
        intrinsics = (
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
        )
        if normalized_camera_info_topic is not None:
            if (
                not normalized_camera_info_topic.startswith("/")
                or any(char.isspace() for char in normalized_camera_info_topic)
            ):
                raise ValueError(
                    "CameraInfo topic must be an absolute ROS name without whitespace"
                )
            if any(value is None for value in intrinsics):
                raise ValueError(
                    "CameraInfo publication requires fx, fy, cx, and cy"
                )
            numeric_intrinsics = tuple(float(value) for value in intrinsics)
            if not all(math.isfinite(value) for value in numeric_intrinsics):
                raise ValueError("CameraInfo intrinsics must be finite")
            if numeric_intrinsics[0] <= 0.0 or numeric_intrinsics[1] <= 0.0:
                raise ValueError("CameraInfo fx and fy must be positive")
        elif any(value is not None for value in intrinsics):
            raise ValueError(
                "camera_info_topic is required when CameraInfo intrinsics are provided"
            )
        self.topic = topic
        self.frame_id = frame_id
        self.node_name = node_name
        self.camera_info_topic = normalized_camera_info_topic
        self._camera_intrinsics = (
            None
            if normalized_camera_info_topic is None
            else tuple(float(value) for value in intrinsics)
        )
        self._minimum_interval = 1.0 / float(max_rate_hz)
        self._last_publish_time = 0.0
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publisher: Any | None = None
        self._image_type: Any | None = None
        self._camera_info_publisher: Any | None = None
        self._camera_info_type: Any | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

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
        camera_info_publisher = (
            node.create_publisher(CameraInfo, self.camera_info_topic, qos)
            if self.camera_info_topic is not None
            else None
        )
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._publisher = publisher
        self._image_type = Image
        self._camera_info_publisher = camera_info_publisher
        self._camera_info_type = CameraInfo if camera_info_publisher is not None else None

    def publish(self, frame: FramePacket) -> None:
        if self._node is None or self._publisher is None or self._image_type is None:
            raise RuntimeError("Camera preview publisher has not been started")
        now = time.perf_counter()
        if now - self._last_publish_time < self._minimum_interval:
            return
        height, width = frame.bgr.shape[:2]
        message = self._image_type()
        message.header.stamp = source_timestamp_or_now(
            self._node,
            frame.source_timestamp,
        )
        message.header.frame_id = self.frame_id
        message.height = height
        message.width = width
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = width * 3
        message.data = frame.bgr.tobytes()
        if (
            self._camera_info_publisher is not None
            and self._camera_info_type is not None
            and self._camera_intrinsics is not None
        ):
            fx, fy, cx, cy = self._camera_intrinsics
            camera_info = self._camera_info_type()
            # Assign the exact same Header object so Image and CameraInfo
            # cannot diverge by even one nanosecond or by frame name.
            camera_info.header = message.header
            camera_info.height = height
            camera_info.width = width
            camera_info.distortion_model = "plumb_bob"
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            camera_info.p = [
                fx,
                0.0,
                cx,
                0.0,
                0.0,
                fy,
                cy,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
            self._camera_info_publisher.publish(camera_info)
        self._publisher.publish(message)
        self._last_publish_time = now

    def close(self) -> None:
        runtime = self._runtime
        if runtime is not None and self._node is not None:
            runtime.remove_node(self._node)
        if self._node is not None and self._publisher is not None:
            self._node.destroy_publisher(self._publisher)
        if self._node is not None and self._camera_info_publisher is not None:
            self._node.destroy_publisher(self._camera_info_publisher)
        if self._node is not None:
            self._node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)
        self._runtime = None
        self._node = None
        self._publisher = None
        self._image_type = None
        self._camera_info_publisher = None
        self._camera_info_type = None
        self._last_publish_time = 0.0
