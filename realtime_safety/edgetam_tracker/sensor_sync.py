from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import time
from typing import Any

import numpy as np

from realtime_safety.edgetam_tracker.models import CloudFrame


_POINT_FIELD_DTYPES: dict[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


@dataclass(slots=True, frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_id: str = ""

    def matrix(self) -> np.ndarray:
        return np.array(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )


@dataclass(slots=True)
class TimestampValidation:
    valid: bool
    reason: str
    span_sec: float
    newest_stamp: float
    oldest_stamp: float


@dataclass(slots=True)
class SensorBundle:
    rgb: Any | None
    depth: Any | None
    camera_info: Any | None
    pointcloud: Any | None
    synchronized: bool
    reason: str = ""


def stamp_to_seconds(stamp_or_message: Any) -> float:
    """Return a ROS header/builtin stamp or numeric timestamp as seconds."""

    value = stamp_or_message
    if hasattr(value, "header"):
        value = value.header.stamp
    if hasattr(value, "stamp"):
        value = value.stamp
    if hasattr(value, "sec") and hasattr(value, "nanosec"):
        return float(value.sec) + float(value.nanosec) * 1e-9
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Unsupported timestamp type: {type(stamp_or_message).__name__}")


def validate_timestamps(
    stamps: list[float] | tuple[float, ...],
    *,
    slop_sec: float,
    now_sec: float | None = None,
    max_data_age_sec: float | None = None,
) -> TimestampValidation:
    finite = np.asarray(stamps, dtype=np.float64)
    if finite.size == 0 or not np.isfinite(finite).all():
        return TimestampValidation(False, "invalid_timestamp", float("inf"), 0.0, 0.0)
    newest = float(np.max(finite))
    oldest = float(np.min(finite))
    span = newest - oldest
    if span > max(float(slop_sec), 0.0):
        return TimestampValidation(False, "timestamps_out_of_sync", span, newest, oldest)
    if (
        now_sec is not None
        and max_data_age_sec is not None
        and max_data_age_sec >= 0.0
        and float(now_sec) - oldest > float(max_data_age_sec)
    ):
        return TimestampValidation(False, "sensor_data_stale", span, newest, oldest)
    if now_sec is not None and newest - float(now_sec) > max(float(slop_sec), 0.0):
        return TimestampValidation(False, "sensor_timestamp_in_future", span, newest, oldest)
    return TimestampValidation(True, "", span, newest, oldest)


def camera_intrinsics_from_info(message: Any) -> CameraIntrinsics:
    k = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    values = np.array((fx, fy, cx, cy), dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError("CameraInfo.K does not contain valid positive fx/fy intrinsics")
    width, height = int(message.width), int(message.height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid CameraInfo size: {width}x{height}")
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        frame_id=str(getattr(message.header, "frame_id", "")),
    )


def image_message_to_array(message: Any) -> np.ndarray:
    """Decode common uncompressed ROS Image encodings without cv_bridge."""

    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}, step={step}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(f"Truncated image buffer: {raw.size} < {required}")
    rows = raw[:required].reshape(height, step)
    encoding = str(message.encoding).lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported color image encoding: {message.encoding}")
    array = rows[:, : width * channels].reshape(height, width, channels)
    if channels == 1:
        return array[..., 0].copy()
    if encoding == "bgr8" or encoding == "8uc3":
        return array[..., ::-1].copy()
    if encoding == "bgra8":
        return array[..., [2, 1, 0, 3]].copy()
    return array.copy()


def depth_message_to_meters(message: Any) -> np.ndarray:
    height, width, step = int(message.height), int(message.width), int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid depth dimensions: {width}x{height}, step={step}")
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        scale = 0.001
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        scale = 1.0
    elif encoding == "64fc1":
        dtype = np.dtype(">f8" if bool(message.is_bigendian) else "<f8")
        scale = 1.0
    else:
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")
    byte_rows = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if byte_rows.size < required:
        raise ValueError(f"Truncated depth buffer: {byte_rows.size} < {required}")
    view = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=byte_rows[:required],
        strides=(step, dtype.itemsize),
    )
    depth = np.asarray(view, dtype=np.float32) * scale
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return depth


def depth_to_cloud(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    stamp: float,
    frame_id: str | None = None,
    rgb: np.ndarray | None = None,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = float("inf"),
) -> CloudFrame:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth image must be HxW")
    height, width = depth.shape
    if (height, width) != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth/CameraInfo shape mismatch: {width}x{height} vs "
            f"{intrinsics.width}x{intrinsics.height}"
        )
    valid = (
        np.isfinite(depth)
        & (depth >= float(minimum_depth_m))
        & (depth <= float(maximum_depth_m))
    )
    vv, uu = np.indices(depth.shape, dtype=np.float32)
    z = depth[valid]
    x = (uu[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (vv[valid] - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    pixels = np.column_stack((uu[valid], vv[valid])).astype(np.int32)
    colors = None
    if rgb is not None:
        rgb_array = np.asarray(rgb)
        if rgb_array.shape[:2] != depth.shape:
            raise ValueError("RGB/depth shape mismatch")
        if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
            raise ValueError("RGB image must be HxWx3 or HxWx4")
        colors = np.asarray(rgb_array[..., :3][valid], dtype=np.uint8)
    return CloudFrame(
        points=points,
        colors=colors,
        pixels_uv=pixels,
        source_indices=np.flatnonzero(valid),
        stamp=float(stamp),
        frame_id=frame_id or intrinsics.frame_id,
        image_shape=(height, width),
    )


def _structured_point_dtype(message: Any) -> np.dtype:
    endian = ">" if bool(message.is_bigendian) else "<"
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    for field in message.fields:
        datatype = int(field.datatype)
        if datatype not in _POINT_FIELD_DTYPES:
            continue
        count = max(int(field.count), 1)
        base = np.dtype(endian + _POINT_FIELD_DTYPES[datatype])
        names.append(str(field.name))
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(int(field.offset))
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PointCloud2 must contain x/y/z fields")
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )


def _decode_packed_rgb(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "f":
        if array.dtype.itemsize != 4:
            raise ValueError("Packed floating RGB must be FLOAT32")
        packed = array.astype(array.dtype, copy=False).view(
            np.dtype(">u4" if array.dtype.byteorder == ">" else "<u4")
        )
    else:
        packed = array.astype(np.uint32, copy=False)
    packed = packed.reshape(-1).astype(np.uint32, copy=False)
    return np.column_stack(
        ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
    ).astype(np.uint8)


def pointcloud2_to_cloud(
    message: Any,
    *,
    image_shape: tuple[int, int] | None = None,
) -> CloudFrame:
    """Convert organized or unordered PointCloud2 while preserving source pixels."""

    height, width = int(message.height), int(message.width)
    point_step, row_step = int(message.point_step), int(message.row_step)
    if height > 0 and width == 0 and point_step > 0 and row_step == 0:
        # A depth frame can legitimately have no finite/cropped returns. Keep
        # that current empty frame instead of leaving consumers on the last
        # non-empty cloud.
        return CloudFrame(
            points=np.empty((0, 3), dtype=np.float32),
            colors=None,
            pixels_uv=None,
            source_indices=np.empty(0, dtype=np.int64),
            stamp=stamp_to_seconds(message),
            frame_id=str(message.header.frame_id),
            image_shape=None,
        )
    if height <= 0 or width <= 0 or point_step <= 0 or row_step < width * point_step:
        raise ValueError(
            f"Invalid PointCloud2 layout: {width}x{height}, "
            f"point_step={point_step}, row_step={row_step}"
        )
    dtype = _structured_point_dtype(message)
    raw = memoryview(message.data)
    required = row_step * height
    if len(raw) < required:
        raise ValueError(f"Truncated PointCloud2 buffer: {len(raw)} < {required}")
    records = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=raw[:required],
        strides=(row_step, point_step),
    )
    points = np.column_stack(
        (
            np.asarray(records["x"]).reshape(-1),
            np.asarray(records["y"]).reshape(-1),
            np.asarray(records["z"]).reshape(-1),
        )
    ).astype(np.float32)
    colors = None
    names = set(dtype.names or ())
    color_name = "rgb" if "rgb" in names else "rgba" if "rgba" in names else None
    if color_name is not None:
        colors = _decode_packed_rgb(np.asarray(records[color_name]).reshape(-1))
    finite = np.isfinite(points).all(axis=1)
    source_indices = np.arange(len(points), dtype=np.int64)
    pixels = None
    effective_shape = None
    if height > 1:
        native_shape = (height, width)
        if image_shape is not None and tuple(image_shape) != native_shape:
            raise ValueError(
                "Organized PointCloud2/image shape mismatch: "
                f"{width}x{height} vs {image_shape[1]}x{image_shape[0]}"
            )
        vv, uu = np.indices((height, width), dtype=np.int32)
        pixels = np.column_stack((uu.reshape(-1), vv.reshape(-1)))
        effective_shape = native_shape
    elif image_shape is not None and int(np.prod(image_shape)) == len(points):
        vv, uu = np.indices(image_shape, dtype=np.int32)
        pixels = np.column_stack((uu.reshape(-1), vv.reshape(-1)))
        effective_shape = tuple(int(value) for value in image_shape)
    return CloudFrame(
        points=points[finite],
        colors=None if colors is None else colors[finite],
        pixels_uv=None if pixels is None else pixels[finite],
        source_indices=source_indices[finite],
        stamp=stamp_to_seconds(message),
        frame_id=str(message.header.frame_id),
        image_shape=effective_shape,
    )


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = x * x + y * y + z * z + w * w
    if not np.isfinite(norm) or norm < 1e-15:
        raise ValueError("Invalid zero/non-finite quaternion")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        (
            (1.0 - yy - zz, xy - wz, xz + wy),
            (xy + wz, 1.0 - xx - zz, yz - wx),
            (xz - wy, yz + wx, 1.0 - xx - yy),
        ),
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray,
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rotation = quaternion_matrix_xyzw(quaternion_xyzw)
    offset = np.asarray(translation, dtype=np.float64).reshape(3)
    return (xyz.astype(np.float64) @ rotation.T + offset).astype(np.float32)


def transform_cloud(
    cloud: CloudFrame,
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
    target_frame: str,
) -> CloudFrame:
    return CloudFrame(
        points=transform_points(cloud.points, translation, quaternion_xyzw),
        colors=cloud.colors,
        pixels_uv=cloud.pixels_uv,
        source_indices=cloud.source_indices,
        stamp=cloud.stamp,
        frame_id=target_frame,
        image_shape=cloud.image_shape,
    )


class RosSensorSynchronizer:
    """Approximate ROS synchronizer with a delayed point-cloud safety fallback.

    message_filters handles normal RGB/depth/cloud fusion. Each geometry message
    is also retained briefly; if another modality disappears, the timer emits a
    point-cloud-only (or depth-only) bundle rather than making obstacles vanish.
    """

    def __init__(
        self,
        node: Any,
        *,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        pointcloud_topic: str,
        queue_size: int,
        slop_sec: float,
        fallback_delay_sec: float,
        callback: Any,
    ) -> None:
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2

        self.node = node
        self.callback = callback
        self.slop_sec = max(float(slop_sec), 0.0)
        self.fallback_delay_sec = max(float(fallback_delay_sec), self.slop_sec)
        self.queue_size = max(int(queue_size), 1)
        self._latest: dict[str, Any] = {}
        self._camera_info_by_stamp: OrderedDict[float, Any] = OrderedDict()
        self._pending_geometry: dict[tuple[str, float], tuple[float, Any]] = {}
        self._processed: dict[tuple[str, float], float] = {}
        self._subscribers: list[Any] = []
        self._subscriber_names: list[str] = []
        self._ats: Any | None = None
        self._camera_info_subscription: Any | None = None

        dynamic: list[tuple[str, Any, str]] = []
        if rgb_topic:
            dynamic.append(("rgb", Image, rgb_topic))
        if depth_topic:
            dynamic.append(("depth", Image, depth_topic))
        if pointcloud_topic:
            dynamic.append(("pointcloud", PointCloud2, pointcloud_topic))
        if not dynamic:
            raise ValueError("At least one RGB, depth, or PointCloud2 topic is required")

        for name, message_type, topic in dynamic:
            subscriber = Subscriber(
                node,
                message_type,
                topic,
                qos_profile=qos_profile_sensor_data,
            )
            subscriber.registerCallback(
                lambda message, source=name: self._on_raw(source, message)
            )
            self._subscribers.append(subscriber)
            self._subscriber_names.append(name)

        if len(self._subscribers) >= 2:
            self._ats = ApproximateTimeSynchronizer(
                self._subscribers,
                queue_size=max(int(queue_size), 1),
                slop=self.slop_sec,
                allow_headerless=False,
            )
            self._ats.registerCallback(self._on_synchronized)

        if camera_info_topic:
            self._camera_info_subscription = node.create_subscription(
                CameraInfo,
                camera_info_topic,
                self._on_camera_info,
                qos_profile_sensor_data,
            )
        self._timer = node.create_timer(
            max(min(self.fallback_delay_sec * 0.5, 0.05), 0.01),
            self._flush_fallbacks,
        )

    def _on_camera_info(self, message: Any) -> None:
        self._latest["camera_info"] = message
        stamp = stamp_to_seconds(message)
        self._camera_info_by_stamp[stamp] = message
        self._camera_info_by_stamp.move_to_end(stamp)
        while len(self._camera_info_by_stamp) > self.queue_size * 2:
            self._camera_info_by_stamp.popitem(last=False)

    def _matching_camera_info(self, reference: Any | None) -> Any | None:
        """Return calibration paired to the reference image timestamp."""

        if reference is None or not self._camera_info_by_stamp:
            return self._latest.get("camera_info")
        stamp = stamp_to_seconds(reference)
        candidates = list(self._camera_info_by_stamp.items())
        matched_stamp, matched = min(
            candidates, key=lambda item: abs(item[0] - stamp)
        )
        if abs(matched_stamp - stamp) <= self.slop_sec:
            return matched
        static = next(
            (message for value, message in candidates if abs(value) <= 1e-12),
            None,
        )
        return static if static is not None else matched

    def _on_raw(self, source: str, message: Any) -> None:
        self._latest[source] = message
        if source in {"pointcloud", "depth"}:
            key = (source, stamp_to_seconds(message))
            self._pending_geometry[key] = (time.monotonic(), message)
        if self._ats is None and source in {"pointcloud", "depth"}:
            self._emit_fallback(source, message, "single_geometry_stream")

    def _on_synchronized(self, *messages: Any) -> None:
        bundle: dict[str, Any] = {}
        for source, message in zip(self._subscriber_names, messages):
            bundle[source] = message
        for source in ("pointcloud", "depth"):
            message = bundle.get(source)
            if message is None:
                continue
            key = (source, stamp_to_seconds(message))
            self._processed[key] = time.monotonic()
            self._pending_geometry.pop(key, None)
        self.callback(
            SensorBundle(
                rgb=bundle.get("rgb"),
                depth=bundle.get("depth"),
                camera_info=self._matching_camera_info(
                    bundle.get("rgb")
                    or bundle.get("depth")
                    or bundle.get("pointcloud")
                ),
                pointcloud=bundle.get("pointcloud"),
                synchronized=True,
            )
        )

    def _flush_fallbacks(self) -> None:
        now = time.monotonic()
        for key, (received_at, message) in tuple(self._pending_geometry.items()):
            if now - received_at < self.fallback_delay_sec:
                continue
            self._pending_geometry.pop(key, None)
            if key in self._processed:
                continue
            self._emit_fallback(key[0], message, "fusion_inputs_missing")
        for key, processed_at in tuple(self._processed.items()):
            if now - processed_at > max(self.fallback_delay_sec * 4.0, 1.0):
                self._processed.pop(key, None)

    def _emit_fallback(self, source: str, message: Any, reason: str) -> None:
        key = (source, stamp_to_seconds(message))
        if key in self._processed:
            return
        self._processed[key] = time.monotonic()
        stamp = key[1]
        rgb = self._latest.get("rgb")
        if rgb is not None and abs(stamp_to_seconds(rgb) - stamp) > self.slop_sec:
            rgb = None
        depth = message if source == "depth" else self._latest.get("depth")
        if depth is not None and abs(stamp_to_seconds(depth) - stamp) > self.slop_sec:
            depth = None
        cloud = message if source == "pointcloud" else self._latest.get("pointcloud")
        if cloud is not None and abs(stamp_to_seconds(cloud) - stamp) > self.slop_sec:
            cloud = None
        self.callback(
            SensorBundle(
                rgb=rgb,
                depth=depth,
                camera_info=self._matching_camera_info(
                    rgb or depth or cloud
                ),
                pointcloud=cloud,
                synchronized=False,
                reason=reason,
            )
        )
