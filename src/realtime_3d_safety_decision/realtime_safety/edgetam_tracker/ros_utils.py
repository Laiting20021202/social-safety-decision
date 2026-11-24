from __future__ import annotations

import colorsys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from realtime_safety.edgetam_tracker.models import (
    MaskObservation,
    ProjectionPrompt,
    TrackEstimate,
    TrackingState,
)


_POINT_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
)


def track_color(track_id: int) -> np.ndarray:
    hue = (int(track_id) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return np.rint(np.array((red, green, blue)) * 255.0).astype(np.uint8)


def pack_rgb_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> tuple[bytes, int]:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if colors is None:
        rgb = np.full((len(finite), 3), 180, dtype=np.uint8)[finite]
    else:
        rgb_all = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        count = min(len(finite), len(rgb_all))
        finite = finite[:count]
        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)[:count][finite]
        rgb = rgb_all[:count][finite]
    packed = np.empty(len(xyz), dtype=_POINT_DTYPE)
    if len(xyz):
        packed["x"], packed["y"], packed["z"] = xyz.T
        packed["rgb"] = (
            rgb[:, 0].astype(np.uint32) << 16
            | rgb[:, 1].astype(np.uint32) << 8
            | rgb[:, 2].astype(np.uint32)
        )
    return packed.tobytes(), len(packed)


def make_pointcloud2(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    header: Any,
    pointcloud_type: Any,
    pointfield_type: Any,
) -> Any:
    data, count = pack_rgb_points(points, colors)
    message = pointcloud_type()
    message.header = header
    message.height = 1
    message.width = count
    message.fields = [
        pointfield_type(
            name="x", offset=0, datatype=pointfield_type.FLOAT32, count=1
        ),
        pointfield_type(
            name="y", offset=4, datatype=pointfield_type.FLOAT32, count=1
        ),
        pointfield_type(
            name="z", offset=8, datatype=pointfield_type.FLOAT32, count=1
        ),
        pointfield_type(
            name="rgb", offset=12, datatype=pointfield_type.UINT32, count=1
        ),
    ]
    message.is_bigendian = False
    message.point_step = _POINT_DTYPE.itemsize
    message.row_step = count * message.point_step
    message.data = data
    message.is_dense = True
    return message


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # Project small PCA numerical errors back onto SO(3).
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1.0
        matrix = u @ vh
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.array((x, y, z, w), dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return quaternion.astype(np.float32)


def make_tracked_obstacle_array(
    tracks: Iterable[TrackEstimate],
    *,
    header: Any,
    obstacle_type: Any,
    array_type: Any,
    point_type: Any,
) -> Any:
    array = array_type()
    array.header = header
    result = []
    for track in tracks:
        item = obstacle_type()
        item.header = header
        item.track_id = int(track.track_id)
        item.tracking_state = track.state.value
        item.pointcloud_quality = track.pointcloud_quality.value
        item.mask_quality = track.mask_quality.value
        item.confidence = float(np.clip(track.confidence, 0.0, 1.0))
        measured_centroid = (
            np.median(track.source_points, axis=0)
            if len(track.source_points)
            else track.position
        )
        (
            item.measured_centroid.x,
            item.measured_centroid.y,
            item.measured_centroid.z,
        ) = map(float, measured_centroid)
        (
            item.filtered_centroid.x,
            item.filtered_centroid.y,
            item.filtered_centroid.z,
        ) = map(float, track.position)
        item.aabb_min.x, item.aabb_min.y, item.aabb_min.z = map(
            float, track.aabb.minimum
        )
        item.aabb_max.x, item.aabb_max.y, item.aabb_max.z = map(
            float, track.aabb.maximum
        )
        item.pose.position.x, item.pose.position.y, item.pose.position.z = map(
            float, track.obb.center
        )
        quaternion = rotation_matrix_to_quaternion_xyzw(track.obb.rotation)
        (
            item.pose.orientation.x,
            item.pose.orientation.y,
            item.pose.orientation.z,
            item.pose.orientation.w,
        ) = map(float, quaternion)
        item.size.x, item.size.y, item.size.z = map(float, track.obb.size)
        item.velocity.x, item.velocity.y, item.velocity.z = map(float, track.velocity)
        (
            item.nearest_point.x,
            item.nearest_point.y,
            item.nearest_point.z,
        ) = map(float, track.nearest_point)
        item.nearest_distance = float(track.nearest_distance)
        item.predicted_positions = []
        for position in track.predicted_positions:
            point = point_type()
            point.x, point.y, point.z = map(float, position)
            item.predicted_positions.append(point)
        item.age_frames = max(int(track.age_frames), 0)
        item.hit_count = max(int(track.hit_count), 0)
        item.missed_frame_count = max(int(track.missed_count), 0)
        item.point_count = max(int(track.point_count), 0)
        measurement_seconds = max(
            float(track.last_measurement_timestamp), 0.0
        )
        item.last_measurement_stamp.sec = int(measurement_seconds)
        item.last_measurement_stamp.nanosec = int(
            round((measurement_seconds - int(measurement_seconds)) * 1e9)
        )
        if item.last_measurement_stamp.nanosec >= 1_000_000_000:
            item.last_measurement_stamp.sec += 1
            item.last_measurement_stamp.nanosec = 0
        item.speed = track.speed
        item.uncertainty_margin = float(max(track.uncertainty_margin, 0.0))
        item.edge_tam_refined = bool(track.edge_tam_refined)
        item.prediction_only = bool(
            track.filter_timestamp
            > track.last_measurement_timestamp + 1e-6
        )
        if hasattr(item, "semantic_class"):
            item.semantic_class = str(track.semantic_class)
        if hasattr(item, "semantic_confirmed"):
            item.semantic_confirmed = bool(track.semantic_confirmed)
        result.append(item)
    array.obstacles = result
    return array


def make_marker_array(
    tracks: Iterable[TrackEstimate],
    *,
    header: Any,
    marker_type: Any,
    marker_array_type: Any,
    point_type: Any,
) -> Any:
    array = marker_array_type()
    clear = marker_type()
    clear.header = header
    clear.action = marker_type.DELETEALL
    array.markers.append(clear)
    for track in tracks:
        color = track_color(track.track_id).astype(np.float32) / 255.0
        alpha = 1.0 if track.state in {TrackingState.CONFIRMED, TrackingState.TENTATIVE} else 0.55
        base_id = int(track.track_id) * 10

        box = marker_type()
        box.header = header
        box.ns = "bounding_boxes"
        box.id = base_id
        box.type = marker_type.CUBE
        box.action = marker_type.ADD
        box.pose.position.x, box.pose.position.y, box.pose.position.z = map(
            float, track.obb.center
        )
        quaternion = rotation_matrix_to_quaternion_xyzw(track.obb.rotation)
        (
            box.pose.orientation.x,
            box.pose.orientation.y,
            box.pose.orientation.z,
            box.pose.orientation.w,
        ) = map(float, quaternion)
        box.scale.x, box.scale.y, box.scale.z = map(
            float, np.maximum(track.obb.size, 0.01)
        )
        box.color.r, box.color.g, box.color.b = map(float, color)
        box.color.a = 0.22 * alpha
        array.markers.append(box)

        label = marker_type()
        label.header = header
        label.ns = "labels"
        label.id = base_id + 1
        label.type = marker_type.TEXT_VIEW_FACING
        label.action = marker_type.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = map(
            float, track.obb.center + np.array((0.0, 0.0, track.obb.size[2] * 0.6 + 0.04))
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.07
        label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, alpha
        label.text = (
            f"#{track.track_id} {track.state.value} "
            f"{track.speed:.2f}m/s c={track.confidence:.2f} "
            f"pc={track.pointcloud_quality.value} mask={track.mask_quality.value}"
        )
        array.markers.append(label)

        nearest = marker_type()
        nearest.header = header
        nearest.ns = "nearest_points"
        nearest.id = base_id + 2
        nearest.type = marker_type.SPHERE
        nearest.action = marker_type.ADD
        (
            nearest.pose.position.x,
            nearest.pose.position.y,
            nearest.pose.position.z,
        ) = map(float, track.nearest_point)
        nearest.pose.orientation.w = 1.0
        nearest.scale.x = nearest.scale.y = nearest.scale.z = 0.035
        nearest.color.r, nearest.color.g, nearest.color.b, nearest.color.a = (
            1.0,
            0.2,
            0.1,
            alpha,
        )
        array.markers.append(nearest)

        centroid = marker_type()
        centroid.header = header
        centroid.ns = "centroids"
        centroid.id = base_id + 5
        centroid.type = marker_type.SPHERE
        centroid.action = marker_type.ADD
        (
            centroid.pose.position.x,
            centroid.pose.position.y,
            centroid.pose.position.z,
        ) = map(float, track.position)
        centroid.pose.orientation.w = 1.0
        centroid.scale.x = centroid.scale.y = centroid.scale.z = 0.045
        centroid.color.r, centroid.color.g, centroid.color.b, centroid.color.a = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            alpha,
        )
        array.markers.append(centroid)

        velocity = marker_type()
        velocity.header = header
        velocity.ns = "velocity"
        velocity.id = base_id + 3
        velocity.type = marker_type.ARROW
        velocity.action = marker_type.ADD
        for position in (track.position, track.position + track.velocity):
            point = point_type()
            point.x, point.y, point.z = map(float, position)
            velocity.points.append(point)
        velocity.scale.x, velocity.scale.y, velocity.scale.z = 0.02, 0.04, 0.06
        velocity.color.r, velocity.color.g, velocity.color.b, velocity.color.a = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            alpha,
        )
        array.markers.append(velocity)

        trajectory = marker_type()
        trajectory.header = header
        trajectory.ns = "predictions"
        trajectory.id = base_id + 4
        trajectory.type = marker_type.LINE_STRIP
        trajectory.action = marker_type.ADD
        for position in np.vstack((track.position, track.predicted_positions)):
            point = point_type()
            point.x, point.y, point.z = map(float, position)
            trajectory.points.append(point)
        trajectory.scale.x = 0.018
        trajectory.color.r, trajectory.color.g, trajectory.color.b, trajectory.color.a = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            0.8 * alpha,
        )
        array.markers.append(trajectory)
    return array


def render_debug_image(
    rgb: np.ndarray,
    tracks: Iterable[TrackEstimate],
    prompts: Mapping[int, ProjectionPrompt],
    masks: Mapping[int, MaskObservation],
    *,
    status_text: str = "",
    hand_detections: Iterable[Any] = (),
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    for detection in hand_detections:
        mask = getattr(detection, "mask", None)
        if mask is not None and np.asarray(mask).shape == image.shape[:2]:
            selected = np.asarray(mask, dtype=bool)
            overlay = np.empty_like(image)
            overlay[...] = (255, 64, 190)
            image[selected] = np.rint(
                0.55 * image[selected].astype(np.float32)
                + 0.45 * overlay[selected].astype(np.float32)
            ).astype(np.uint8)
        box = np.rint(np.asarray(getattr(detection, "bbox_xyxy"))).astype(int)
        if box.shape == (4,):
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 64, 190), 2)
            cv2.putText(
                image,
                "MEDIAPIPE HAND",
                (max(x1, 0), max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 64, 190),
                1,
                cv2.LINE_AA,
            )
    for track in tracks:
        color_rgb = track_color(track.track_id)
        mask_observation = masks.get(track.track_id)
        if mask_observation is not None and mask_observation.mask.shape == image.shape[:2]:
            mask = mask_observation.mask
            overlay = np.broadcast_to(color_rgb, image.shape)
            image[mask] = np.rint(
                0.55 * image[mask].astype(np.float32)
                + 0.45 * overlay[mask].astype(np.float32)
            ).astype(np.uint8)
        prompt = prompts.get(track.track_id)
        if prompt is not None:
            x1, y1, x2, y2 = np.rint(prompt.box_xyxy).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb.tolist(), 2)
            for point in np.rint(prompt.positive_points).astype(int):
                cv2.circle(image, tuple(point), 3, (0, 255, 0), -1)
            if prompt.negative_points is not None:
                for point in np.rint(prompt.negative_points).astype(int):
                    cv2.circle(image, tuple(point), 3, (255, 40, 40), -1)
            text_origin = (max(x1, 0), max(y1 - 6, 14))
        else:
            text_origin = (8, 20 + 18 * (track.track_id % 20))
        cv2.putText(
            image,
            (
                f"#{track.track_id} {track.state.value} {track.speed:.2f}m/s "
                f"c={track.confidence:.2f} pc={track.pointcloud_quality.value} "
                f"mask={track.mask_quality.value}"
                + (
                    f" RP:{prompt.reason[:28]}"
                    if prompt is not None and prompt.re_prompt
                    else ""
                )
            ),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color_rgb.tolist(),
            1,
            cv2.LINE_AA,
        )
    if status_text:
        cv2.putText(
            image,
            status_text,
            (8, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 220, 40),
            1,
            cv2.LINE_AA,
        )
    return image


def make_rgb_image_message(
    rgb: np.ndarray,
    *,
    header: Any,
    image_type: Any,
) -> Any:
    array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., :3])
    message = image_type()
    message.header = header
    message.height, message.width = array.shape[:2]
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = int(message.width * 3)
    message.data = array.tobytes()
    return message
