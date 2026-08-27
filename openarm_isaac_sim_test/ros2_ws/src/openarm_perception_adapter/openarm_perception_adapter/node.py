from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker

from openarm_sim.camera_math import PinholeIntrinsics, back_project_depth


@dataclass
class SyncedInputs:
    color: Image | None = None
    depth: Image | None = None
    mask: Image | None = None
    camera_info: CameraInfo | None = None


class RgbdMaskAdapter(Node):
    """Convert a YOLO/SAM binary mask plus aligned depth to a metric obstacle cloud.

    The default formal path requires an external ``/perception/hand_mask``. The
    optional HSV fallback is deliberately disabled and labeled non-formal.
    """

    def __init__(self) -> None:
        super().__init__("openarm_rgbd_mask_adapter")
        self.declare_parameter("input_mode", "mask")
        self.declare_parameter("mask_topic", "/perception/hand_mask")
        self.declare_parameter(
            "obstacle_cloud_topic", "/edgetam_tracker/obstacle_cloud"
        )
        self.declare_parameter("allow_hsv_placeholder", False)
        self.declare_parameter("voxel_stride", 2)
        self.declare_parameter("timeout_sec", 0.25)
        self.allow_hsv = bool(self.get_parameter("allow_hsv_placeholder").value)
        self.input_mode = str(self.get_parameter("input_mode").value).strip().lower()
        if self.input_mode not in {"mask", "social_cloud"}:
            raise ValueError("input_mode must be mask or social_cloud")
        self.inputs: dict[tuple[int, int], SyncedInputs] = {}
        self.latest_info: CameraInfo | None = None
        self.last_output_ns = self.get_clock().now().nanoseconds
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/perception/obstacles", qos_profile_sensor_data
        )
        self.marker_pub = self.create_publisher(Marker, "/perception/obstacle_marker", 10)
        self.events_pub = self.create_publisher(String, "/openarm/events", 50)
        if self.input_mode == "social_cloud":
            source = str(self.get_parameter("obstacle_cloud_topic").value)
            if source == "/perception/obstacles":
                raise ValueError("social obstacle input and adapter output must differ")
            self.create_subscription(
                PointCloud2,
                source,
                self._social_cloud,
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"social-safety obstacle cloud input: {source}")
        else:
            self.create_subscription(
                Image, "/rgbd/color/image_raw", self._color, qos_profile_sensor_data
            )
            self.create_subscription(
                Image,
                "/rgbd/aligned_depth_to_color/image_raw",
                self._depth,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo, "/rgbd/color/camera_info", self._info, qos_profile_sensor_data
            )
            if not self.allow_hsv:
                self.create_subscription(
                    Image,
                    str(self.get_parameter("mask_topic").value),
                    self._mask,
                    qos_profile_sensor_data,
                )
        self.create_timer(0.05, self._timeout_check)

    def _social_cloud(self, message: PointCloud2) -> None:
        """Relay only the perception system's measured obstacle geometry."""

        self.cloud_pub.publish(message)
        points = _xyz_points(message)
        header = Header(stamp=message.header.stamp, frame_id=message.header.frame_id)
        self._publish_marker(points, header)
        self.events_pub.publish(
            String(data=f"perception_output,{len(points)},social_safety")
        )
        self.last_output_ns = self.get_clock().now().nanoseconds

    def _color(self, message: Image) -> None:
        key = _stamp_key(message)
        self.inputs.setdefault(key, SyncedInputs()).color = message
        self._try_process(key)

    def _depth(self, message: Image) -> None:
        key = _stamp_key(message)
        self.inputs.setdefault(key, SyncedInputs()).depth = message
        self._try_process(key)

    def _mask(self, message: Image) -> None:
        key = _stamp_key(message)
        self.inputs.setdefault(key, SyncedInputs()).mask = message
        self._try_process(key)

    def _info(self, message: CameraInfo) -> None:
        self.latest_info = message

    def _try_process(self, key: tuple[int, int]) -> None:
        bundle = self.inputs[key]
        if bundle.color is None or bundle.depth is None or self.latest_info is None:
            return
        if bundle.mask is None and not self.allow_hsv:
            return
        color = _decode_color(bundle.color)
        depth = _decode_depth(bundle.depth)
        mask = _decode_mask(bundle.mask) if bundle.mask is not None else _skin_hsv_placeholder(color)
        if mask.shape != depth.shape:
            self.get_logger().error("mask and aligned depth dimensions differ")
            del self.inputs[key]
            return
        info = self.latest_info
        intrinsics = PinholeIntrinsics(
            width=int(info.width),
            height=int(info.height),
            fx=float(info.k[0]),
            fy=float(info.k[4]),
            cx=float(info.k[2]),
            cy=float(info.k[5]),
        )
        masked_depth = np.where(mask, depth, np.nan).astype(np.float32)
        points, _ = back_project_depth(masked_depth, intrinsics, near_clip=0.1, far_clip=3.0)
        stride = max(1, int(self.get_parameter("voxel_stride").value))
        points = points[::stride]
        header = Header(stamp=bundle.depth.header.stamp, frame_id=bundle.depth.header.frame_id)
        cloud = point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.cloud_pub.publish(cloud)
        self._publish_marker(points, header)
        self.events_pub.publish(
            String(data=f"perception_output,{len(points)},{'hsv_placeholder' if self.allow_hsv else 'external_mask'}")
        )
        self.last_output_ns = self.get_clock().now().nanoseconds
        del self.inputs[key]
        for old_key in sorted(self.inputs)[:-5]:
            self.inputs.pop(old_key, None)

    def _publish_marker(self, points: np.ndarray, header: Header) -> None:
        if len(points) == 0:
            return
        lower = np.quantile(points, 0.02, axis=0)
        upper = np.quantile(points, 0.98, axis=0)
        marker = Marker()
        marker.header = header
        marker.ns = "perception_obstacle"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        center = (lower + upper) / 2.0
        size = np.maximum(upper - lower, 0.025)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, center)
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, size)
        marker.color.r = 1.0
        marker.color.g = 0.05
        marker.color.b = 0.2
        marker.color.a = 0.85
        self.marker_pub.publish(marker)

    def _timeout_check(self) -> None:
        timeout_ns = int(float(self.get_parameter("timeout_sec").value) * 1e9)
        if self.get_clock().now().nanoseconds - self.last_output_ns > timeout_ns:
            self.events_pub.publish(String(data="perception_timeout"))
            self.last_output_ns = self.get_clock().now().nanoseconds


def _stamp_key(message: Image) -> tuple[int, int]:
    return message.header.stamp.sec, message.header.stamp.nanosec


def _xyz_points(message: PointCloud2) -> np.ndarray:
    rows = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    names = getattr(getattr(rows, "dtype", None), "names", None)
    if names:
        return np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
            np.float32, copy=False
        )
    return np.asarray(list(rows), dtype=np.float32).reshape(-1, 3)


def _decode_color(message: Image) -> np.ndarray:
    if message.encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported color encoding: {message.encoding}")
    image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
    return image[..., ::-1] if message.encoding == "bgr8" else image


def _decode_depth(message: Image) -> np.ndarray:
    if message.encoding != "32FC1":
        raise ValueError(f"depth must be 32FC1 meter, got {message.encoding}")
    return np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width)


def _decode_mask(message: Image) -> np.ndarray:
    if message.encoding not in {"mono8", "8UC1"}:
        raise ValueError(f"mask must be mono8/8UC1, got {message.encoding}")
    return np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width) > 0


def _skin_hsv_placeholder(rgb: np.ndarray) -> np.ndarray:
    # Simple normalized-RGB rule for bring-up only; never reported as YOLO/SAM.
    values = rgb.astype(np.float32) + 1.0
    total = values.sum(axis=2)
    red = values[..., 0] / total
    green = values[..., 1] / total
    blue = values[..., 2] / total
    return (red > 0.38) & (green > 0.23) & (green < 0.38) & (blue < 0.30)


def main() -> None:
    rclpy.init()
    node = RgbdMaskAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
