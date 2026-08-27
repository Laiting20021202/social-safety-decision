from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _rotation_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _project_root() -> Path:
    value = os.environ.get("OPENARM_SIM_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _camera_transform() -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    root = _project_root()
    camera = yaml.safe_load((root / "config/camera.yaml").read_text())["camera"]
    scene = yaml.safe_load((root / "config/scene.yaml").read_text())
    explicit_pose = camera.get("world_pose")
    if explicit_pose:
        position = np.asarray(explicit_pose["position"], dtype=np.float32)
        roll, pitch, yaw = (
            math.radians(float(value)) for value in explicit_pose["rpy_deg"]
        )
    else:
        workspace = np.asarray(scene["zones"]["workspace"]["center"], dtype=np.float32)
        target = workspace + np.asarray(
            camera.get("aim_offset", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        position = workspace + np.array(
            [
                -float(camera["horizontal_offset_to_workspace_center"]),
                float(camera["lateral_offset"]),
                float(camera["height_above_table"]),
            ],
            dtype=np.float32,
        )
        direction = target - position
        direction /= np.linalg.norm(direction)
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        pitch = math.atan2(float(-direction[2]), float(np.linalg.norm(direction[:2])))
        roll = 0.0
    world_from_link = _rotation_from_rpy(roll, pitch, yaw)
    optical_to_link = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    pointcloud = camera["pointcloud"]
    return (
        position,
        world_from_link @ optical_to_link,
        int(pointcloud["pixel_stride"]),
        np.asarray(pointcloud["world_crop_min"], dtype=np.float32),
        np.asarray(pointcloud["world_crop_max"], dtype=np.float32),
    )


class GazeboRgbdAdapter(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_rgbd_adapter")
        (
            self._position,
            self._world_from_optical,
            self._stride,
            self._crop_min,
            self._crop_max,
        ) = _camera_transform()
        self._input_cloud_count = 0
        self._output_cloud_count = 0
        self._camera_pose_received = False
        self._pose_history: deque[tuple[int, np.ndarray, np.ndarray]] = deque(maxlen=300)
        self._optical_to_link = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            # RELIABLE publisher is compatible with both RViz's reliable
            # display and sensor-data BEST_EFFORT consumers. KEEP_LAST/depth=1
            # still prevents a slow viewer from accumulating stale frames.
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._aligned_pub = self.create_publisher(
            Image, "/rgbd/aligned_depth_to_color/image_raw", output_qos
        )
        self._world_pub = self.create_publisher(
            PointCloud2, "/rgbd/points_world", output_qos
        )
        self.create_subscription(
            Image, "/rgbd/depth/image_raw", self._depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            "/rgbd/points",
            self._points_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.create_subscription(
            PoseStamped, "/sim/camera/pose", self._camera_pose_callback, 10
        )
        self.get_logger().info(
            f"Gazebo RGB-D adapter ready; world cloud stride={self._stride}"
        )

    def _camera_pose_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id not in {"", "world"}:
            return
        position = message.pose.position
        orientation = message.pose.orientation
        self._position = np.asarray(
            [position.x, position.y, position.z], dtype=np.float32
        )
        self._world_from_optical = _rotation_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        ) @ self._optical_to_link
        stamp_ns = _stamp_ns(message.header.stamp)
        self._pose_history.append(
            (stamp_ns, self._position.copy(), self._world_from_optical.copy())
        )
        if not self._camera_pose_received:
            self._camera_pose_received = True
            self.get_logger().info(
                "Using live Gazebo camera pose for world point-cloud projection"
            )

    def _depth_callback(self, message: Image) -> None:
        message.header.frame_id = "rgbd_color_optical_frame"
        self._aligned_pub.publish(message)

    def _points_callback(self, message: PointCloud2) -> None:
        self._input_cloud_count += 1
        field_offsets = {field.name: field.offset for field in message.fields}
        if not {"x", "y", "z"}.issubset(field_offsets):
            self.get_logger().error("input PointCloud2 is missing x/y/z fields")
            return
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.row_step)
        packed_rows = rows[:, : message.width * message.point_step]
        points = np.ascontiguousarray(packed_rows).reshape(-1, message.point_step)

        grid = np.arange(message.height * message.width).reshape(
            message.height, message.width
        )
        indexes = grid[
            :: max(1, self._stride), :: max(1, self._stride)
        ].reshape(-1)
        # Decode only the spatial display sample.  The old implementation
        # decoded all 307,200 XYZ tuples before selecting, monopolizing the
        # single ROS executor long enough to skip subsequent depth frames.
        selected = points[indexes]

        def float_field(data: np.ndarray, name: str) -> np.ndarray:
            offset = field_offsets[name]
            return np.ascontiguousarray(data[:, offset : offset + 4]).view("<f4").reshape(-1)

        xyz = np.column_stack(
            (
                float_field(selected, "x"),
                float_field(selected, "y"),
                float_field(selected, "z"),
            )
        )
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
        xyz = xyz[valid]
        selected = selected[valid]
        position, world_from_optical = self._pose_for_stamp(
            _stamp_ns(message.header.stamp)
        )
        world_xyz = xyz @ world_from_optical.T + position
        inside = np.logical_and(world_xyz >= self._crop_min, world_xyz <= self._crop_max).all(axis=1)
        world_xyz = world_xyz[inside]
        selected = selected[inside]
        output = np.empty(
            len(world_xyz),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")],
        )
        output["x"], output["y"], output["z"] = world_xyz.T
        if "rgb" in field_offsets:
            offset = field_offsets["rgb"]
            output["rgb"] = np.ascontiguousarray(
                selected[:, offset : offset + 4]
            ).view("<u4").reshape(-1)
        else:
            output["rgb"] = np.uint32(0x00FFFFFF)
        result = PointCloud2()
        result.header = message.header
        result.header.frame_id = "world"
        result.height = 1
        result.width = int(len(world_xyz))
        result.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        result.is_bigendian = False
        result.point_step = 16
        result.row_step = result.width * result.point_step
        result.data = output.tobytes()
        result.is_dense = True
        self._world_pub.publish(result)
        self._output_cloud_count += 1
        if self._output_cloud_count == 1:
            self.get_logger().info(
                f"Published first world cloud: {result.width} points frame={result.header.frame_id}"
            )

    def _pose_for_stamp(self, stamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the camera pose nearest to, but not newer than, a depth frame."""

        if not self._pose_history:
            return self._position, self._world_from_optical
        for pose_stamp, position, rotation in reversed(self._pose_history):
            if pose_stamp <= stamp_ns:
                return position, rotation
        # During startup the first cloud can precede the first model-state
        # update.  The oldest live pose is safer than mixing in a future edit.
        _, position, rotation = self._pose_history[0]
        return position, rotation


def _stamp_ns(stamp: object) -> int:
    return int(getattr(stamp, "sec")) * 1_000_000_000 + int(
        getattr(stamp, "nanosec")
    )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboRgbdAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Humble can surface a take_message conversion error while launch is
        # simultaneously shutting down the DDS context.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
