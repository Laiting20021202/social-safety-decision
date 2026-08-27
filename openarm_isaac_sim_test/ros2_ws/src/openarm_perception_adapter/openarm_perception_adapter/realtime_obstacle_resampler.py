from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener


_OPENARM_SELF_FILTER_FRAMES = [
    *(f"openarm_left_link{index}" for index in range(8)),
    "openarm_left_hand",
    "openarm_left_left_finger",
    "openarm_left_right_finger",
    *(f"openarm_right_link{index}" for index in range(8)),
    "openarm_right_hand",
    "openarm_right_left_finger",
    "openarm_right_right_finger",
]
_OPENARM_SELF_FILTER_RADII_M = [
    0.080,
    0.115,
    0.094,
    0.186,
    0.119,
    0.136,
    0.076,
    0.099,
    0.085,
    0.093,
    0.093,
] * 2
_OPENARM_FRAMES_PER_SIDE = 11
_OPENARM_SELF_FILTER_SEGMENTS = [
    # link0 -> ... -> link7 -> hand
    *((index, index + 1) for index in range(8)),
    # hand -> each parallel finger (do not connect finger-to-finger).
    (8, 9),
    (8, 10),
    *((
        first + _OPENARM_FRAMES_PER_SIDE,
        second + _OPENARM_FRAMES_PER_SIDE,
    ) for first, second in (
        *((index, index + 1) for index in range(8)),
        (8, 9),
        (8, 10),
    )),
]


@dataclass
class _RgbdFrame:
    stamp_ns: int
    xyz: np.ndarray
    height: int
    width: int
    gray: np.ndarray | None = None
    processed: bool = False


def _stamp_ns(message: object) -> int:
    stamp = message.header.stamp  # type: ignore[attr-defined]
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _xyz_array(message: PointCloud2) -> np.ndarray:
    rows = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=False
    )
    names = getattr(getattr(rows, "dtype", None), "names", None)
    if names:
        return np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
            np.float32, copy=False
        )
    return np.asarray(list(rows), dtype=np.float32).reshape(-1, 3)


def metric_spatial_gate(
    xyz: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Select finite points inside a model-authorized metric 3D extent."""

    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    low = np.asarray(lower, dtype=np.float32).reshape(3)
    high = np.asarray(upper, dtype=np.float32).reshape(3)
    return np.isfinite(points).all(axis=1) & np.logical_and(
        points >= low,
        points <= high,
    ).all(axis=1)


def mask_metric_support(
    mask: np.ndarray,
    xyz: np.ndarray,
    *,
    model_depth: float,
    depth_half_width: float,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> int:
    """Count current RGB-D samples supporting a tracked image-space mask."""

    selected = np.asarray(mask).reshape(-1).astype(bool)
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    selected &= np.isfinite(points).all(axis=1)
    if np.isfinite(model_depth):
        selected &= np.abs(points[:, 2] - float(model_depth)) <= float(
            depth_half_width
        )
    if lower is not None and upper is not None:
        selected &= metric_spatial_gate(points, lower, upper)
    return int(np.count_nonzero(selected))


def filter_robot_self_points(
    xyz: np.ndarray,
    centers: np.ndarray,
    radii_m: np.ndarray,
    *,
    padding_m: float = 0.0,
    segment_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int]:
    """Remove samples inside measured link spheres and articulated capsules."""

    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    link_centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    radii = np.asarray(radii_m, dtype=np.float32).reshape(-1)
    if len(link_centers) != len(radii):
        raise ValueError("robot self-filter centers and radii must have equal length")
    if not len(points) or not len(link_centers):
        return points.copy(), 0
    inside = np.zeros(len(points), dtype=bool)
    for center, radius in zip(link_centers, radii, strict=True):
        distance = points - center
        limit = max(float(radius) + float(padding_m), 0.0)
        inside |= np.einsum("ij,ij->i", distance, distance) <= limit * limit
    for first, second in segment_pairs or ():
        if not (0 <= first < len(link_centers) and 0 <= second < len(link_centers)):
            raise ValueError("robot self-filter segment index is out of range")
        start = link_centers[first]
        end = link_centers[second]
        delta = end - start
        length_squared = float(np.dot(delta, delta))
        if length_squared <= 1e-12:
            continue
        projection = np.clip(
            ((points - start) @ delta) / length_squared,
            0.0,
            1.0,
        )
        closest = start + projection[:, None] * delta
        distance = points - closest
        # The smaller endpoint sphere gives the connecting link a useful
        # conservative thickness without turning the larger joint housing
        # into a capsule that could erase a nearby real hand wholesale.
        limit = max(
            min(float(radii[first]), float(radii[second]))
            + float(padding_m),
            0.0,
        )
        inside |= np.einsum("ij,ij->i", distance, distance) <= limit * limit
    return points[~inside], int(np.count_nonzero(inside))


def robot_dominated_candidate(
    original_count: int,
    retained_count: int,
    *,
    minimum_retained_fraction: float,
) -> bool:
    """Return true when a neural candidate is almost entirely OpenArm.

    Sphere/capsule filtering can leave a thin silhouette around a moving link.
    Those few pixels must not seed optical-flow tracking as a human hand.
    """

    if original_count <= 0:
        return False
    threshold = float(np.clip(minimum_retained_fraction, 0.0, 1.0))
    return retained_count / original_count < threshold


def nearest_timestamp(
    stamps: list[int] | tuple[int, ...],
    target_ns: int,
    tolerance_ns: int,
) -> int | None:
    """Return the nearest timestamp inside an inclusive synchronization window."""

    candidates = [
        int(stamp)
        for stamp in stamps
        if abs(int(stamp) - int(target_ns)) <= int(tolerance_ns)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda stamp: abs(stamp - int(target_ns)))


def seed_mask_from_model(
    xyz: np.ndarray,
    model_points: np.ndarray,
    shape: tuple[int, int],
    *,
    padding_m: float = 0.012,
) -> np.ndarray:
    """Recover an image-space seed from a model-confirmed 3D cloud."""

    values = np.asarray(model_points, dtype=np.float32).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 8:
        return np.zeros(shape, dtype=np.uint8)

    points = np.asarray(xyz, dtype=np.float32).reshape(shape[0], shape[1], 3)
    projected = _project_model_points(points, values)
    if projected is not None:
        mask = np.zeros(shape, dtype=np.uint8)
        for column, row in projected:
            cv2.circle(mask, (int(column), int(row)), 4, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        if int(np.count_nonzero(mask)) >= 8:
            return mask

    # AABB is only a fallback for malformed/non-pinhole organized clouds.
    # Projection above prevents unrelated table pixels inside the hand's 3D
    # bounding box from turning into one oversized obstacle mask.
    lower = np.quantile(values, 0.005, axis=0) - float(padding_m)
    upper = np.quantile(values, 0.995, axis=0) + float(padding_m)
    flattened = points.reshape(-1, 3)
    selected = np.isfinite(flattened).all(axis=1)
    selected &= np.logical_and(flattened >= lower, flattened <= upper).all(axis=1)
    mask = selected.reshape(shape).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _project_model_points(
    organized_xyz: np.ndarray,
    model_points: np.ndarray,
) -> np.ndarray | None:
    """Project optical-frame XYZ with intrinsics fitted from the pixel grid."""

    height, width = organized_xyz.shape[:2]
    stride = max(int(np.sqrt((height * width) / 5000.0)), 1)
    sampled = organized_xyz[::stride, ::stride]
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    depth = sampled[..., 2]
    valid = np.isfinite(sampled).all(axis=2) & (depth > 1e-4)
    if int(np.count_nonzero(valid)) < 32:
        return None
    x_ratio = sampled[..., 0][valid] / depth[valid]
    y_ratio = sampled[..., 1][valid] / depth[valid]
    horizontal = np.column_stack((x_ratio, np.ones_like(x_ratio)))
    vertical = np.column_stack((y_ratio, np.ones_like(y_ratio)))
    fx, cx = np.linalg.lstsq(horizontal, columns[valid], rcond=None)[0]
    fy, cy = np.linalg.lstsq(vertical, rows[valid], rcond=None)[0]
    if (
        not np.isfinite((fx, fy, cx, cy)).all()
        or not 10.0 <= abs(float(fx)) <= 10000.0
        or not 10.0 <= abs(float(fy)) <= 10000.0
    ):
        return None
    horizontal_error = np.median(np.abs(horizontal @ (fx, cx) - columns[valid]))
    vertical_error = np.median(np.abs(vertical @ (fy, cy) - rows[valid]))
    if max(float(horizontal_error), float(vertical_error)) > 1.5:
        return None

    depth = model_points[:, 2]
    valid_model = np.isfinite(model_points).all(axis=1) & (depth > 1e-4)
    if int(np.count_nonzero(valid_model)) < 8:
        return None
    values = model_points[valid_model]
    projected_columns = np.rint(fx * values[:, 0] / values[:, 2] + cx)
    projected_rows = np.rint(fy * values[:, 1] / values[:, 2] + cy)
    inside = (
        (projected_columns >= 0)
        & (projected_columns < width)
        & (projected_rows >= 0)
        & (projected_rows < height)
    )
    if int(np.count_nonzero(inside)) < 8:
        return None
    return np.column_stack(
        (projected_columns[inside], projected_rows[inside])
    ).astype(np.int32)


def track_mask(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Translate a confirmed mask with sparse RGB optical flow."""

    if not np.any(mask):
        return mask.copy()
    features = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=mask,
        maxCorners=160,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
    )
    if features is None or len(features) < 4:
        return mask.copy()
    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            24,
            0.01,
        ),
    )
    if moved is None or status is None:
        return mask.copy()
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= errors.reshape(-1) < 35.0
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        moved,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            24,
            0.01,
        ),
    )
    if returned is None or backward_status is None:
        return mask.copy()
    valid &= backward_status.reshape(-1).astype(bool)
    round_trip = np.linalg.norm(
        returned.reshape(-1, 2) - features.reshape(-1, 2), axis=1
    )
    valid &= round_trip < 2.0
    if int(np.count_nonzero(valid)) < 4:
        return mask.copy()
    displacement = moved.reshape(-1, 2)[valid] - features.reshape(-1, 2)[valid]
    dx, dy = np.median(displacement, axis=0)
    if not np.isfinite((dx, dy)).all() or abs(float(dx)) > 40 or abs(float(dy)) > 40:
        return mask.copy()
    transform = np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32)
    return cv2.warpAffine(
        mask,
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class RealtimeObstacleResampler(Node):
    """Back-project a real model seed against every fresh simulator RGB-D frame.

    The slow neural output remains the only source allowed to create or reset
    an obstacle mask. Between neural frames, sparse RGB optical flow moves that
    mask and the current PointCloud2 supplies new metric XYZ samples. No
    simulator entity pose or ground-truth collision geometry is consumed.
    """

    def __init__(self) -> None:
        super().__init__("openarm_realtime_obstacle_resampler")
        self.declare_parameter(
            "model_cloud_topic", "/realtime_safety/yolo_obstacles/pointcloud"
        )
        self.declare_parameter("rgb_topic", "/rgbd/color/image_raw")
        self.declare_parameter("rgbd_cloud_topic", "/rgbd/points")
        self.declare_parameter(
            "output_cloud_topic", "/edgetam_tracker/obstacle_cloud_realtime"
        )
        self.declare_parameter("model_timeout_sec", 2.0)
        # The detector follows the 30 Hz RGB stream while the organized cloud
        # is normally 15 Hz.  A model result can therefore arrive before its
        # depth frame, or carry the adjacent RGB stamp.  Buffer it instead of
        # silently dropping a valid hand detection.
        self.declare_parameter("model_sync_tolerance_sec", 0.15)
        self.declare_parameter("maximum_buffer_frames", 48)
        self.declare_parameter("depth_half_width_m", 0.05)
        self.declare_parameter("spatial_padding_m", 0.025)
        self.declare_parameter("spatial_recovery_max_age_sec", 0.40)
        self.declare_parameter("empty_model_clear_updates", 2)
        # The GUI can select either EdgeTAM or the retained MediaPipe/YOLO
        # backend. Apply the OpenArm TF geometry filter after that mux so both
        # paths are unable to publish the robot itself as a human obstacle.
        self.declare_parameter("self_filter.enabled", True)
        self.declare_parameter(
            "self_filter.link_frames", _OPENARM_SELF_FILTER_FRAMES
        )
        self.declare_parameter(
            "self_filter.link_radii_m", _OPENARM_SELF_FILTER_RADII_M
        )
        self.declare_parameter("self_filter.padding_m", 0.005)
        self.declare_parameter("self_filter.minimum_retained_fraction", 0.15)
        self.declare_parameter("self_filter.tf_timeout_sec", 0.01)
        self._timeout_ns = int(
            float(self.get_parameter("model_timeout_sec").value) * 1e9
        )
        self._sync_tolerance_ns = max(
            int(float(self.get_parameter("model_sync_tolerance_sec").value) * 1e9),
            0,
        )
        self._maximum_frames = max(
            int(self.get_parameter("maximum_buffer_frames").value), 8
        )
        self._depth_half_width = max(
            float(self.get_parameter("depth_half_width_m").value), 0.03
        )
        self._spatial_padding = max(
            float(self.get_parameter("spatial_padding_m").value), 0.005
        )
        self._spatial_recovery_max_age_ns = max(
            int(
                float(
                    self.get_parameter("spatial_recovery_max_age_sec").value
                )
                * 1e9
            ),
            0,
        )
        self._empty_model_clear_updates = max(
            int(self.get_parameter("empty_model_clear_updates").value), 1
        )
        self._self_filter_enabled = bool(
            self.get_parameter("self_filter.enabled").value
        )
        self._self_filter_frames = [
            str(value)
            for value in self.get_parameter("self_filter.link_frames").value
        ]
        self._self_filter_radii = np.asarray(
            self.get_parameter("self_filter.link_radii_m").value,
            dtype=np.float32,
        )
        if len(self._self_filter_frames) != len(self._self_filter_radii):
            raise ValueError(
                "self_filter.link_frames and link_radii_m must have equal length"
            )
        self._self_filter_padding = max(
            float(self.get_parameter("self_filter.padding_m").value), 0.0
        )
        self._self_filter_minimum_retained_fraction = float(
            np.clip(
                self.get_parameter(
                    "self_filter.minimum_retained_fraction"
                ).value,
                0.0,
                1.0,
            )
        )
        self._self_filter_tf_timeout = max(
            float(self.get_parameter("self_filter.tf_timeout_sec").value), 0.0
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._self_filter_removed = 0
        self._self_filter_status = "disabled"
        self._frames: OrderedDict[int, _RgbdFrame] = OrderedDict()
        self._images: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pending_models: OrderedDict[int, np.ndarray] = OrderedDict()
        self._mask: np.ndarray | None = None
        self._mask_gray: np.ndarray | None = None
        self._mask_stamp_ns = 0
        self._model_stamp_ns = 0
        self._model_depth = float("nan")
        self._model_lower: np.ndarray | None = None
        self._model_upper: np.ndarray | None = None
        self._empty_model_updates = 0
        self._model_received = 0
        self._model_applied = 0
        self._model_seed_rejected = 0
        self._geometry_empty_frames = 0
        self._spatial_recoveries = 0
        self._flow_rejections = 0
        self._transient_geometry_skips = 0
        self._published = 0
        self._last_status_ns = 0

        output = str(self.get_parameter("output_cloud_topic").value)
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            PointCloud2, output, output_qos
        )
        self._status_publisher = self.create_publisher(
            String, output + "/status", 10
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("rgb_topic").value),
            self._receive_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("rgbd_cloud_topic").value),
            self._receive_rgbd,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("model_cloud_topic").value),
            self._receive_model,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Fresh RGB-D obstacle resampler: "
            f"model={self.get_parameter('model_cloud_topic').value} "
            f"output={output}"
        )

    def _receive_rgb(self, message: Image) -> None:
        if message.encoding not in {"rgb8", "bgr8"}:
            return
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), int(message.width), 3
        )
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY if message.encoding == "rgb8" else cv2.COLOR_BGR2GRAY,
        )
        stamp = _stamp_ns(message)
        self._images[stamp] = gray
        frame = self._frames.get(stamp)
        if frame is not None and frame.gray is None:
            frame.gray = gray
            self._apply_pending_model(frame)
            self._process_frame(frame)
        self._trim_buffers()

    def _receive_rgbd(self, message: PointCloud2) -> None:
        if int(message.height) <= 1 or int(message.width) <= 1:
            self.get_logger().error("RGB-D source cloud must remain organized")
            return
        stamp = _stamp_ns(message)
        xyz = _xyz_array(message)
        expected = int(message.height) * int(message.width)
        if len(xyz) != expected:
            self.get_logger().error("Organized RGB-D cloud has an invalid point count")
            return
        frame = _RgbdFrame(
            stamp_ns=stamp,
            xyz=xyz,
            height=int(message.height),
            width=int(message.width),
            gray=self._images.get(stamp),
        )
        self._frames[stamp] = frame
        if frame.gray is not None:
            self._apply_pending_model(frame)
            self._process_frame(frame)
        self._trim_buffers()

    def _receive_model(self, message: PointCloud2) -> None:
        self._model_received += 1
        points = _xyz_array(message)
        points = points[np.isfinite(points).all(axis=1)]
        points = self._filter_robot_points(
            points,
            str(message.header.frame_id) or "rgbd_color_optical_frame",
        )
        if len(points) < 8:
            # One neural miss is tolerated to avoid visual flicker. Two
            # consecutive measured empty outputs mean the hand has left the
            # RGB frame; clear the old optical-flow mask before it becomes a
            # stale obstacle at the image boundary.
            self._empty_model_updates += 1
            if self._empty_model_updates >= self._empty_model_clear_updates:
                self._clear_mask()
            return
        self._empty_model_updates = 0
        stamp = _stamp_ns(message)
        frame_stamp = nearest_timestamp(
            tuple(
                key
                for key, candidate in self._frames.items()
                if candidate.gray is not None
            ),
            stamp,
            # Do not bind a newly arrived RGB detection to the previous 15 Hz
            # depth frame.  Its exact cloud often arrives a few callbacks
            # later; queue below and let that newer frame consume it.
            min(self._sync_tolerance_ns, 2_000_000),
        )
        frame = self._frames.get(frame_stamp) if frame_stamp is not None else None
        if frame is None or frame.gray is None:
            self._pending_models[stamp] = points
            self._trim_buffers()
            return
        self._apply_model_points(frame, points)

    def _apply_pending_model(self, frame: _RgbdFrame) -> None:
        stamp = nearest_timestamp(
            tuple(self._pending_models),
            frame.stamp_ns,
            self._sync_tolerance_ns,
        )
        if stamp is None:
            return
        points = self._pending_models.pop(stamp)
        self._apply_model_points(frame, points)

    def _apply_model_points(
        self, frame: _RgbdFrame, points: np.ndarray
    ) -> None:
        mask = seed_mask_from_model(
            frame.xyz, points, (frame.height, frame.width)
        )
        if int(np.count_nonzero(mask)) < 8:
            self._model_seed_rejected += 1
            return
        self._model_applied += 1
        self._mask = mask
        self._mask_gray = frame.gray
        self._mask_stamp_ns = frame.stamp_ns
        # Model validity begins at the RGB-D frame actually used to recover
        # its mask, not at a potentially adjacent 30 Hz RGB timestamp.
        self._model_stamp_ns = frame.stamp_ns
        self._model_depth = float(np.median(points[:, 2]))
        self._model_lower = (
            np.quantile(points, 0.005, axis=0) - self._spatial_padding
        ).astype(np.float32)
        self._model_upper = (
            np.quantile(points, 0.995, axis=0) + self._spatial_padding
        ).astype(np.float32)
        for newer in tuple(self._frames.values()):
            if newer.stamp_ns > frame.stamp_ns and newer.gray is not None:
                self._advance_mask(newer)

    def _advance_mask(self, frame: _RgbdFrame) -> None:
        if self._mask is None or self._mask_gray is None or frame.gray is None:
            return
        if frame.stamp_ns <= self._mask_stamp_ns:
            return
        tracked = track_mask(self._mask_gray, frame.gray, self._mask)
        old_support = mask_metric_support(
            self._mask,
            frame.xyz,
            model_depth=self._model_depth,
            depth_half_width=self._depth_half_width,
            lower=self._model_lower,
            upper=self._model_upper,
        )
        tracked_support = mask_metric_support(
            tracked,
            frame.xyz,
            model_depth=self._model_depth,
            depth_half_width=self._depth_half_width,
            lower=self._model_lower,
            upper=self._model_upper,
        )
        if old_support > tracked_support:
            self._flow_rejections += 1
        else:
            self._mask = tracked
        self._mask_gray = frame.gray
        self._mask_stamp_ns = frame.stamp_ns

    def _process_frame(self, frame: _RgbdFrame) -> None:
        if frame.processed:
            return
        frame.processed = True
        self._advance_mask(frame)
        points = np.empty((0, 3), dtype=np.float32)
        model_age_ns = frame.stamp_ns - self._model_stamp_ns
        if (
            self._mask is not None
            and model_age_ns >= 0
            and model_age_ns <= self._timeout_ns
            and self._mask.shape == (frame.height, frame.width)
        ):
            mask_selected = self._mask.reshape(-1).astype(bool)
            mask_selected &= np.isfinite(frame.xyz).all(axis=1)
            selected = mask_selected.copy()
            # The neural cloud is the only authority allowed to seed geometry.
            # Constrain optical-flow mask pixels to its recent metric 3D extent
            # so a drifting 2D mask cannot absorb a same-depth table patch.
            if self._model_lower is not None and self._model_upper is not None:
                selected &= metric_spatial_gate(
                    frame.xyz, self._model_lower, self._model_upper
                )
            if np.isfinite(self._model_depth):
                depth_selected = mask_selected & (
                    np.abs(frame.xyz[:, 2] - self._model_depth)
                    <= self._depth_half_width
                )
                selected &= depth_selected
                # The hand may move beyond the previous sparse model AABB
                # before the next neural result.  For a short, model-authorized
                # window recover from the tracked 2D mask plus metric depth;
                # robot self filtering still runs below.  This keeps fresh
                # hand geometry alive without accepting an unbounded scene.
                if (
                    int(np.count_nonzero(selected)) < 8
                    and model_age_ns <= self._spatial_recovery_max_age_ns
                    and int(np.count_nonzero(depth_selected)) >= 8
                ):
                    selected = depth_selected
                    self._spatial_recoveries += 1
            points = frame.xyz[selected]
            points = self._filter_robot_points(
                points, "rgbd_color_optical_frame"
            )
            if len(points) >= 8:
                self._model_depth = float(
                    0.85 * self._model_depth + 0.15 * np.median(points[:, 2])
                )
                # Move the authorized metric envelope with the measured hand
                # instead of freezing it at the last neural frame.
                self._model_lower = (
                    np.quantile(points, 0.005, axis=0) - self._spatial_padding
                ).astype(np.float32)
                self._model_upper = (
                    np.quantile(points, 0.995, axis=0) + self._spatial_padding
                ).astype(np.float32)
            else:
                points = np.empty((0, 3), dtype=np.float32)
                self._geometry_empty_frames += 1
        elif model_age_ns > self._timeout_ns:
            self._clear_mask()

        # A failed optical-flow/depth intersection is not evidence that the
        # measured hand disappeared.  Do not flash an empty cloud for this
        # single-frame processing miss; the next fresh RGB-D frame or neural
        # seed will replace the last valid sample.  Confirmed model misses or
        # timeout clear the mask above and still publish an explicit empty
        # cloud, so stale geometry cannot persist indefinitely.
        if (
            not len(points)
            and self._mask is not None
            and 0 <= model_age_ns <= self._timeout_ns
        ):
            self._transient_geometry_skips += 1
            return

        header = Header()
        header.stamp.sec = int(frame.stamp_ns // 1_000_000_000)
        header.stamp.nanosec = int(frame.stamp_ns % 1_000_000_000)
        header.frame_id = "rgbd_color_optical_frame"
        self._publisher.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))
        self._published += 1
        if frame.stamp_ns - self._last_status_ns >= 1_000_000_000:
            age = max(model_age_ns, 0) * 1e-9
            self._status_publisher.publish(
                String(
                    data=(
                        f"fresh_rgbd frame={self._published} points={len(points)} "
                        f"model_age={age:.3f}s ground_truth=false "
                        f"model={self._model_applied}/{self._model_received} "
                        f"seed_rejected={self._model_seed_rejected} "
                        f"pending={len(self._pending_models)} "
                        f"geometry_empty={self._geometry_empty_frames} "
                        f"spatial_recovery={self._spatial_recoveries} "
                        f"flow_rejected={self._flow_rejections} "
                        f"transient_skips={self._transient_geometry_skips} "
                        f"self_filter={self._self_filter_status} "
                        f"self_removed={self._self_filter_removed}"
                    )
                )
            )
            self._last_status_ns = frame.stamp_ns

    def _filter_robot_points(
        self, points: np.ndarray, target_frame: str
    ) -> np.ndarray:
        if not self._self_filter_enabled or not len(points):
            self._self_filter_status = "disabled" if not self._self_filter_enabled else "empty"
            return points
        centers: list[list[float]] = []
        radii: list[float] = []
        timeout = Duration(seconds=self._self_filter_tf_timeout)
        for frame, radius in zip(
            self._self_filter_frames, self._self_filter_radii, strict=True
        ):
            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame,
                    frame,
                    rclpy.time.Time(),
                    timeout=timeout,
                )
            except TransformException:
                continue
            translation = transform.transform.translation
            centers.append([translation.x, translation.y, translation.z])
            radii.append(float(radius))
        if len(centers) != len(self._self_filter_frames):
            # Preserve the candidate on incomplete TF rather than hiding a
            # real person. The status makes this fail-safe degradation visible.
            self._self_filter_status = (
                f"unavailable:{len(centers)}/{len(self._self_filter_frames)}"
            )
            return points
        filtered, removed = filter_robot_self_points(
            points,
            np.asarray(centers, dtype=np.float32),
            np.asarray(radii, dtype=np.float32),
            padding_m=self._self_filter_padding,
            segment_pairs=_OPENARM_SELF_FILTER_SEGMENTS,
        )
        self._self_filter_removed += removed
        if robot_dominated_candidate(
            len(points),
            len(filtered),
            minimum_retained_fraction=(
                self._self_filter_minimum_retained_fraction
            ),
        ):
            self._self_filter_status = (
                f"rejected_self:{len(filtered)}/{len(points)}"
            )
            return np.empty((0, 3), dtype=np.float32)
        self._self_filter_status = f"active:{len(filtered)}/{len(points)}"
        return filtered

    def _clear_mask(self) -> None:
        self._mask = None
        self._mask_gray = None
        self._mask_stamp_ns = 0
        self._model_stamp_ns = 0
        self._model_depth = float("nan")
        self._model_lower = None
        self._model_upper = None
        self._empty_model_updates = 0

    def _trim_buffers(self) -> None:
        while len(self._frames) > self._maximum_frames:
            self._frames.popitem(last=False)
        while len(self._images) > self._maximum_frames:
            self._images.popitem(last=False)
        while len(self._pending_models) > self._maximum_frames:
            self._pending_models.popitem(last=False)
        if self._frames:
            newest_frame = next(reversed(self._frames))
            expired = [
                stamp
                for stamp in self._pending_models
                if newest_frame - stamp > self._sync_tolerance_ns
            ]
            for stamp in expired:
                self._pending_models.pop(stamp, None)


def main() -> None:
    rclpy.init()
    node = RealtimeObstacleResampler()
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
