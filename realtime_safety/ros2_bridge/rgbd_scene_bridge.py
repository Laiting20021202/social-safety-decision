from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from realtime_safety.edgetam_tracker.sensor_sync import (
    camera_intrinsics_from_info,
    depth_message_to_meters,
    image_message_to_array,
    pointcloud2_to_cloud,
    quaternion_matrix_xyzw,
    stamp_to_seconds,
    validate_timestamps,
)
from realtime_safety.ros2_bridge.runtime import (
    acquire_ros2_runtime,
    release_ros2_runtime,
)
from realtime_safety.types import PointCloudFrame


LOGGER = logging.getLogger(__name__)


_OPTICAL_TO_LINK = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(slots=True, frozen=True)
class RgbdProjectionConfig:
    """Limits for image-driven metric RGB-D reconstruction."""

    max_points: int = 24_000
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 4.0
    sync_slop_sec: float = 0.02
    depth_noise_stddev_m: float = 0.0
    depth_noise_clip_sigma: float = 3.0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_points < 100:
            raise ValueError("max_points must be at least 100")
        if self.minimum_depth_m < 0.0:
            raise ValueError("minimum_depth_m cannot be negative")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        if self.sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec cannot be negative")
        if self.depth_noise_stddev_m < 0.0:
            raise ValueError("depth_noise_stddev_m cannot be negative")
        if self.depth_noise_clip_sigma <= 0.0:
            raise ValueError("depth_noise_clip_sigma must be positive")


@dataclass(slots=True)
class ProjectedRgbdFrame:
    """One current RGB-D image reconstructed in optical and world frames."""

    pipeline_cloud: PointCloudFrame
    optical_points: np.ndarray
    colors: np.ndarray
    world_points: np.ndarray
    world_cloud: PointCloudFrame


class RgbdFrameProjector:
    """Back-project synchronized RGB, depth and CameraInfo without sim clouds."""

    def __init__(
        self,
        config: RgbdProjectionConfig | None = None,
        *,
        camera_position: np.ndarray | None = None,
        world_from_optical: np.ndarray | None = None,
        world_crop_min: np.ndarray | None = None,
        world_crop_max: np.ndarray | None = None,
    ) -> None:
        self.config = config or RgbdProjectionConfig()
        self._rng = np.random.default_rng(self.config.noise_seed)
        self._camera_position = np.zeros(3, dtype=np.float64)
        self._world_from_optical = np.eye(3, dtype=np.float64)
        self._ray_cache_key: tuple[int, int, float, float, float, float] | None = None
        self._ray_x: np.ndarray | None = None
        self._ray_y: np.ndarray | None = None
        self._world_crop_min = (
            None
            if world_crop_min is None
            else np.asarray(world_crop_min, dtype=np.float64).reshape(3)
        )
        self._world_crop_max = (
            None
            if world_crop_max is None
            else np.asarray(world_crop_max, dtype=np.float64).reshape(3)
        )
        if (self._world_crop_min is None) != (self._world_crop_max is None):
            raise ValueError("world crop requires both minimum and maximum")
        if (
            self._world_crop_min is not None
            and np.any(self._world_crop_max <= self._world_crop_min)
        ):
            raise ValueError("world crop maximum must exceed minimum")
        self.set_camera_transform(
            np.zeros(3) if camera_position is None else camera_position,
            np.eye(3) if world_from_optical is None else world_from_optical,
        )

    def set_camera_transform(
        self, camera_position: np.ndarray, world_from_optical: np.ndarray
    ) -> None:
        position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        rotation = np.asarray(world_from_optical, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("camera transform must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
            raise ValueError("world_from_optical must be orthonormal")
        self._camera_position = position.copy()
        self._world_from_optical = rotation.copy()

    def project(
        self,
        rgb_message: Any,
        depth_message: Any,
        camera_info: Any,
        *,
        frame_index: int,
        camera_position: np.ndarray | None = None,
        world_from_optical: np.ndarray | None = None,
    ) -> ProjectedRgbdFrame:
        rgb_stamp = stamp_to_seconds(rgb_message)
        depth_stamp = stamp_to_seconds(depth_message)
        info_stamp = stamp_to_seconds(camera_info)
        stamps = [rgb_stamp, depth_stamp]
        # A zero-stamped CameraInfo is a valid static calibration. A live
        # CameraInfo must belong to the same acquisition as RGB and depth.
        if info_stamp > 0.0:
            stamps.append(info_stamp)
        validation = validate_timestamps(
            stamps, slop_sec=self.config.sync_slop_sec
        )
        if not validation.valid:
            raise ValueError(f"RGB-D timestamps rejected: {validation.reason}")

        rgb = image_message_to_array(rgb_message)
        depth = depth_message_to_meters(depth_message)
        intrinsics = camera_intrinsics_from_info(camera_info)
        if rgb.shape[:2] != depth.shape:
            raise ValueError("RGB and aligned depth dimensions differ")
        if (intrinsics.height, intrinsics.width) != depth.shape:
            raise ValueError("CameraInfo and aligned depth dimensions differ")
        if self.config.depth_noise_stddev_m > 0.0:
            finite = np.isfinite(depth)
            noise = self._rng.normal(
                0.0, self.config.depth_noise_stddev_m, depth.shape
            ).astype(np.float32)
            limit = (
                self.config.depth_noise_clip_sigma
                * self.config.depth_noise_stddev_m
            )
            np.clip(noise, -limit, limit, out=noise)
            depth = depth.copy()
            depth[finite] += noise[finite]

        # Construct the aligned pointmap directly from cached pinhole rays.
        # The generic depth_to_cloud helper intentionally returns compact
        # points/pixel indices, which would then need another 307k-pixel pass
        # to restore the dense map used by RGB masks.  Caching the rays keeps
        # the image-driven production path comfortably above 10 Hz at 640x480.
        ray_x, ray_y = self._projection_rays(intrinsics)
        valid = (
            np.isfinite(depth)
            & (depth >= self.config.minimum_depth_m)
            & (depth <= self.config.maximum_depth_m)
        )
        dense_optical = np.empty((*depth.shape, 3), dtype=np.float32)
        dense_optical[..., 0] = ray_x * depth
        dense_optical[..., 1] = ray_y * depth
        dense_optical[..., 2] = depth
        dense_optical[~valid] = np.nan
        dense_internal = np.stack(
            (
                dense_optical[..., 0],
                dense_optical[..., 2],
                -dense_optical[..., 1],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        sampled_optical, sampled_colors = _sample_dense_current_frame(
            dense_optical, rgb[..., :3], self.config.max_points
        )
        sampled_internal = np.column_stack(
            (
                sampled_optical[:, 0],
                sampled_optical[:, 2],
                -sampled_optical[:, 1],
            )
        ).astype(np.float32, copy=False)

        position = (
            self._camera_position
            if camera_position is None
            else np.asarray(camera_position, dtype=np.float64).reshape(3)
        )
        rotation = (
            self._world_from_optical
            if world_from_optical is None
            else np.asarray(world_from_optical, dtype=np.float64).reshape(3, 3)
        )
        world_points = (
            sampled_optical.astype(np.float64) @ rotation.T + position
        ).astype(np.float32)
        world_colors = sampled_colors
        if self._world_crop_min is not None:
            inside = np.logical_and(
                world_points >= self._world_crop_min,
                world_points <= self._world_crop_max,
            ).all(axis=1)
            world_points, world_colors = world_points[inside], world_colors[inside]

        count = len(sampled_internal)
        pipeline_cloud = PointCloudFrame(
            points=sampled_internal,
            colors=sampled_colors,
            confidence=np.ones(count, dtype=np.float32),
            pointmap=dense_internal,
            frame_index=int(frame_index),
            timestamp=float(depth_stamp),
            anchor_frame_index=int(frame_index),
            inference_ms=0.0,
            valid=count > 0,
            source="rgbd_depth_backprojection",
            metric_scale=1.0,
        )
        world_count = len(world_points)
        world_cloud = PointCloudFrame(
            points=world_points,
            colors=world_colors,
            confidence=np.ones(world_count, dtype=np.float32),
            pointmap=np.empty((0, 0, 3), dtype=np.float32),
            frame_index=int(frame_index),
            timestamp=float(depth_stamp),
            anchor_frame_index=int(frame_index),
            inference_ms=0.0,
            valid=world_count > 0,
            source="rgbd_depth_backprojection_world",
            metric_scale=1.0,
        )
        return ProjectedRgbdFrame(
            pipeline_cloud=pipeline_cloud,
            optical_points=sampled_optical,
            colors=sampled_colors,
            world_points=world_points,
            world_cloud=world_cloud,
        )

    def _projection_rays(self, intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
        key = (
            int(intrinsics.height),
            int(intrinsics.width),
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
        )
        if key != self._ray_cache_key:
            height, width, fx, fy, cx, cy = key
            self._ray_x = (
                (np.arange(width, dtype=np.float32) - cx) / fx
            ).reshape(1, width)
            self._ray_y = (
                (np.arange(height, dtype=np.float32) - cy) / fy
            ).reshape(height, 1)
            self._ray_cache_key = key
        assert self._ray_x is not None and self._ray_y is not None
        return self._ray_x, self._ray_y


def _sample_dense_current_frame(
    pointmap: np.ndarray, rgb: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use a stable pixel grid so occlusion never shifts historical samples."""

    height, width = pointmap.shape[:2]
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max_points))))
    points = pointmap[::stride, ::stride].reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8)[::stride, ::stride].reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    return points[valid], colors[valid]


class RgbdSceneBridge:
    """Consume simulator metric clouds without running monocular depth."""

    def __init__(
        self,
        raw_topic: str,
        world_topic: str,
        on_raw_cloud: Callable[[PointCloudFrame], None],
        on_debug_cloud: Callable[[np.ndarray, np.ndarray, bool, float], None] | None,
        *,
        max_points: int = 24_000,
    ) -> None:
        self.raw_topic = raw_topic
        self.world_topic = world_topic
        self.on_raw_cloud = on_raw_cloud
        self.on_debug_cloud = on_debug_cloud
        self.max_points = max(int(max_points), 100)
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._frame_index = 0
        self._lock = threading.Lock()
        self._last_received = {"raw": 0.0, "world": 0.0}

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2

        runtime = acquire_ros2_runtime()
        node = Node("realtime_safety_rgbd_scene_bridge", context=runtime.context)
        node.create_subscription(
            PointCloud2,
            self.raw_topic,
            lambda message: self._receive(message, world=False),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            PointCloud2,
            self.world_topic,
            lambda message: self._receive(message, world=True),
            qos_profile_sensor_data,
        )
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        LOGGER.info(
            "Simulator RGB-D scene bridge raw=%s world=%s max_points=%d",
            self.raw_topic,
            self.world_topic,
            self.max_points,
        )

    def _receive(self, message: Any, *, world: bool) -> None:
        try:
            cloud = pointcloud2_to_cloud(message)
            points = np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)
            colors = (
                np.full((len(points), 3), 190, dtype=np.uint8)
                if cloud.colors is None
                else np.asarray(cloud.colors, dtype=np.uint8).reshape(-1, 3)
            )
            if len(points) > self.max_points:
                step = int(np.ceil(len(points) / self.max_points))
                points, colors = points[::step], colors[::step]
            debug_colors = colors
            if world:
                # The physical table is intentionally dark. Lift only the
                # debug-layer luminance so its metric surface stays visible
                # against Viser's black background.
                debug_colors = np.clip(
                    colors.astype(np.float32) * 0.65 + 70.0, 0.0, 255.0
                ).astype(np.uint8)
            now = time.monotonic()
            key = "world" if world else "raw"
            previous = self._last_received[key]
            self._last_received[key] = now
            rate = 0.0 if previous <= 0.0 else 1.0 / max(now - previous, 1e-6)
            if self.on_debug_cloud is not None:
                self.on_debug_cloud(points, debug_colors, world, rate)
            if world:
                return
            # RealtimePipeline's established internal coordinates are
            # x-right/y-forward/z-up; ROS optical is x-right/y-down/z-forward.
            internal_points = np.column_stack(
                (points[:, 0], points[:, 2], -points[:, 1])
            ).astype(np.float32, copy=False)
            with self._lock:
                frame_index = self._frame_index
                self._frame_index += 1
            count = len(points)
            frame = PointCloudFrame(
                points=internal_points,
                colors=colors,
                confidence=np.ones(count, dtype=np.float32),
                pointmap=np.empty((0, 0, 3), dtype=np.float32),
                frame_index=frame_index,
                timestamp=float(cloud.stamp),
                anchor_frame_index=frame_index,
                inference_ms=0.0,
                valid=count > 0,
                source="simulator_rgbd",
                metric_scale=1.0,
            )
            self.on_raw_cloud(frame)
        except Exception:
            LOGGER.exception("Could not consume simulator %s point cloud", "world" if world else "raw")

    def close(self) -> None:
        runtime, node = self._runtime, self._node
        self._runtime = None
        self._node = None
        if runtime is None or node is None:
            return
        runtime.remove_node(node)
        node.destroy_node()
        release_ros2_runtime(runtime)


class RgbdImageSceneBridge:
    """Generate current-frame point clouds from synchronized RGB-D images.

    Unlike :class:`RgbdSceneBridge`, this bridge never subscribes to a
    simulator-provided PointCloud2.  RGB supplies color, aligned depth supplies
    metric geometry, and CameraInfo supplies the pinhole intrinsics.  A camera
    pose is used only to express those measured rays in the world frame.
    """

    def __init__(
        self,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        on_raw_cloud: Callable[[PointCloudFrame], None],
        on_debug_cloud: Callable[[np.ndarray, np.ndarray, bool, float], None]
        | None,
        *,
        camera_pose_topic: str = "",
        on_world_cloud: Callable[[PointCloudFrame], None] | None = None,
        projection_config: RgbdProjectionConfig | None = None,
        camera_position: np.ndarray | None = None,
        world_from_optical: np.ndarray | None = None,
        world_crop_min: np.ndarray | None = None,
        world_crop_max: np.ndarray | None = None,
        pose_is_optical: bool = False,
    ) -> None:
        self.color_topic = str(color_topic)
        self.depth_topic = str(depth_topic)
        self.camera_info_topic = str(camera_info_topic)
        self.camera_pose_topic = str(camera_pose_topic)
        self.on_raw_cloud = on_raw_cloud
        self.on_debug_cloud = on_debug_cloud
        self.on_world_cloud = on_world_cloud
        self.pose_is_optical = bool(pose_is_optical)
        self.projector = RgbdFrameProjector(
            projection_config,
            camera_position=camera_position,
            world_from_optical=world_from_optical,
            world_crop_min=world_crop_min,
            world_crop_max=world_crop_max,
        )
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscribers: list[Any] = []
        self._synchronizer: Any | None = None
        self._frame_index = 0
        self._lock = threading.Lock()
        self._pose_history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
            maxlen=300
        )
        self._last_received = 0.0
        # Viser recreates its bounded point-cloud handle to guarantee that an
        # occluded pixel disappears immediately.  That WebSocket/WebGL update
        # can take longer than one camera period, so it must never run in the
        # ROS image callback.  This one-slot mailbox deliberately drops stale
        # display work while the ROS publishers still receive every current
        # reconstructed frame.
        self._debug_condition = threading.Condition()
        self._debug_pending: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
        ] | None = None
        self._debug_stop = threading.Event()
        self._debug_thread: threading.Thread | None = None
        self._diagnostic_started_at = 0.0
        self._diagnostic_frames = 0
        self._diagnostic_processing_sec = 0.0
        self._diagnostic_projection_sec = 0.0
        self._diagnostic_pipeline_sec = 0.0

    def start(self) -> None:
        if self._node is not None:
            return
        from geometry_msgs.msg import PoseStamped
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        runtime = acquire_ros2_runtime()
        node = Node(
            "realtime_safety_rgbd_image_scene_bridge", context=runtime.context
        )
        self._subscribers = [
            Subscriber(
                node,
                Image,
                self.color_topic,
                qos_profile=qos_profile_sensor_data,
            ),
            Subscriber(
                node,
                Image,
                self.depth_topic,
                qos_profile=qos_profile_sensor_data,
            ),
            Subscriber(
                node,
                CameraInfo,
                self.camera_info_topic,
                qos_profile=qos_profile_sensor_data,
            ),
        ]
        self._synchronizer = ApproximateTimeSynchronizer(
            self._subscribers,
            queue_size=5,
            slop=self.projector.config.sync_slop_sec,
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._receive_synchronized)
        if self.camera_pose_topic:
            node.create_subscription(
                PoseStamped,
                self.camera_pose_topic,
                self._receive_camera_pose,
                qos_profile_sensor_data,
            )
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        if self.on_debug_cloud is not None:
            self._debug_stop.clear()
            self._debug_thread = threading.Thread(
                target=self._debug_worker,
                name="realtime-safety-rgbd-debug-renderer",
                daemon=True,
            )
            self._debug_thread.start()
        LOGGER.info(
            "Image-driven RGB-D scene bridge color=%s depth=%s info=%s "
            "pose=%s max_points=%d depth_noise_stddev_m=%.6f",
            self.color_topic,
            self.depth_topic,
            self.camera_info_topic,
            self.camera_pose_topic or "configured_extrinsics",
            self.projector.config.max_points,
            self.projector.config.depth_noise_stddev_m,
        )

    def update_camera_pose(
        self,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
        *,
        stamp: float,
        pose_is_optical: bool | None = None,
    ) -> None:
        """Install a timestamped camera extrinsic from an external pose source."""

        rotation = quaternion_matrix_xyzw(quaternion_xyzw)
        optical = self.pose_is_optical if pose_is_optical is None else pose_is_optical
        world_from_optical = rotation if optical else rotation @ _OPTICAL_TO_LINK
        values = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.isfinite(values).all():
            raise ValueError("camera pose position must be finite")
        with self._lock:
            self._pose_history.append(
                (float(stamp), values.copy(), world_from_optical.copy())
            )

    def _receive_camera_pose(self, message: Any) -> None:
        frame_id = str(getattr(message.header, "frame_id", ""))
        if frame_id not in {"", "world"}:
            LOGGER.warning("Ignoring camera pose in non-world frame %s", frame_id)
            return
        position = message.pose.position
        orientation = message.pose.orientation
        try:
            self.update_camera_pose(
                np.array((position.x, position.y, position.z), dtype=np.float64),
                np.array(
                    (orientation.x, orientation.y, orientation.z, orientation.w),
                    dtype=np.float64,
                ),
                stamp=stamp_to_seconds(message),
            )
        except (TypeError, ValueError):
            LOGGER.exception("Could not accept live camera pose")

    def _pose_for_stamp(self, stamp: float) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            if not self._pose_history:
                return None, None
            for pose_stamp, position, rotation in reversed(self._pose_history):
                if pose_stamp <= stamp:
                    return position.copy(), rotation.copy()
            _, position, rotation = self._pose_history[0]
            return position.copy(), rotation.copy()

    def _receive_synchronized(
        self, rgb_message: Any, depth_message: Any, camera_info: Any
    ) -> None:
        processing_started = time.perf_counter()
        try:
            stamp = stamp_to_seconds(depth_message)
            position, rotation = self._pose_for_stamp(stamp)
            with self._lock:
                frame_index = self._frame_index
                self._frame_index += 1
            projection_started = time.perf_counter()
            projected = self.projector.project(
                rgb_message,
                depth_message,
                camera_info,
                frame_index=frame_index,
                camera_position=position,
                world_from_optical=rotation,
            )
            projection_sec = time.perf_counter() - projection_started
            now = time.monotonic()
            previous = self._last_received
            self._last_received = now
            rate = 0.0 if previous <= 0.0 else 1.0 / max(now - previous, 1e-6)
            pipeline_started = time.perf_counter()
            self.on_raw_cloud(projected.pipeline_cloud)
            pipeline_sec = time.perf_counter() - pipeline_started
            if self.on_world_cloud is not None:
                self.on_world_cloud(projected.world_cloud)
            if self.on_debug_cloud is not None:
                world_colors = np.clip(
                    projected.world_cloud.colors.astype(np.float32) * 0.65 + 70.0,
                    0.0,
                    255.0,
                ).astype(np.uint8)
                pending = (
                    projected.optical_points,
                    projected.colors,
                    projected.world_points,
                    world_colors,
                    rate,
                )
                if self._debug_thread is None:
                    # Direct embedded/unit-test use does not call start().
                    self._render_debug_frame(pending)
                else:
                    with self._debug_condition:
                        self._debug_pending = pending
                        self._debug_condition.notify()
            self._record_diagnostics(
                time.perf_counter() - processing_started,
                projection_sec=projection_sec,
                pipeline_sec=pipeline_sec,
            )
        except Exception:
            LOGGER.exception("Could not reconstruct synchronized RGB-D images")

    def _record_diagnostics(
        self,
        processing_sec: float,
        *,
        projection_sec: float,
        pipeline_sec: float,
    ) -> None:
        now = time.monotonic()
        if self._diagnostic_started_at <= 0.0:
            self._diagnostic_started_at = now
        self._diagnostic_frames += 1
        self._diagnostic_processing_sec += float(processing_sec)
        self._diagnostic_projection_sec += float(projection_sec)
        self._diagnostic_pipeline_sec += float(pipeline_sec)
        elapsed = now - self._diagnostic_started_at
        if elapsed < 2.0:
            return
        LOGGER.info(
            "Image-driven RGB-D diagnostics callback_rate=%.2fHz "
            "processing_avg=%.2fms projection_avg=%.2fms "
            "pipeline_avg=%.2fms latest_only_debug=%s",
            self._diagnostic_frames / max(elapsed, 1e-9),
            1000.0 * self._diagnostic_processing_sec
            / max(self._diagnostic_frames, 1),
            1000.0 * self._diagnostic_projection_sec
            / max(self._diagnostic_frames, 1),
            1000.0 * self._diagnostic_pipeline_sec
            / max(self._diagnostic_frames, 1),
            self._debug_thread is not None,
        )
        self._diagnostic_started_at = now
        self._diagnostic_frames = 0
        self._diagnostic_processing_sec = 0.0
        self._diagnostic_projection_sec = 0.0
        self._diagnostic_pipeline_sec = 0.0

    def _debug_worker(self) -> None:
        """Render only the newest RGB-D cloud without blocking ROS input."""

        while not self._debug_stop.is_set():
            with self._debug_condition:
                while self._debug_pending is None and not self._debug_stop.is_set():
                    self._debug_condition.wait(timeout=0.5)
                if self._debug_stop.is_set():
                    return
                pending = self._debug_pending
                self._debug_pending = None
            if pending is None or self.on_debug_cloud is None:
                continue
            try:
                self._render_debug_frame(pending)
            except Exception:
                LOGGER.exception("Could not render the current RGB-D debug cloud")

    def _render_debug_frame(
        self,
        pending: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
        ],
    ) -> None:
        if self.on_debug_cloud is None:
            return
        optical_points, optical_colors, world_points, world_colors, rate = pending
        self.on_debug_cloud(optical_points, optical_colors, False, rate)
        self.on_debug_cloud(world_points, world_colors, True, rate)

    def close(self) -> None:
        runtime, node = self._runtime, self._node
        self._runtime = None
        self._node = None
        self._subscribers = []
        self._synchronizer = None
        self._debug_stop.set()
        with self._debug_condition:
            self._debug_pending = None
            self._debug_condition.notify_all()
        debug_thread = self._debug_thread
        self._debug_thread = None
        if debug_thread is not None:
            debug_thread.join(timeout=3.0)
        if runtime is None or node is None:
            return
        runtime.remove_node(node)
        node.destroy_node()
        release_ros2_runtime(runtime)
