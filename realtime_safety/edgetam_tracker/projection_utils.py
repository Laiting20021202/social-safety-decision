from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from realtime_safety.edgetam_tracker.models import (
    Cluster3D,
    ProjectionPrompt,
    TrackEstimate,
)


@dataclass(slots=True)
class ProjectionConfig:
    minimum_depth: float = 0.02
    maximum_depth: float | None = None
    box_padding_pixels: int = 6
    box_padding_ratio: float = 0.05
    projection_dilation_pixels: int = 3
    maximum_positive_points: int = 6
    minimum_positive_spacing_pixels: float = 8.0
    positive_boundary_margin_pixels: float = 1.0
    density_blur_pixels: int = 9
    generate_negative_points: bool = True
    maximum_negative_points: int = 4

    def __post_init__(self) -> None:
        if self.minimum_depth < 0.0:
            raise ValueError("minimum_depth cannot be negative")
        if self.maximum_positive_points < 1:
            raise ValueError("maximum_positive_points must be positive")
        if self.projection_dilation_pixels < 0:
            raise ValueError("projection_dilation_pixels cannot be negative")


@dataclass(frozen=True, slots=True)
class ProjectedPoints:
    uv: np.ndarray
    depth: np.ndarray
    source_indices: np.ndarray
    camera_points: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "uv",
            np.asarray(self.uv, dtype=np.float32).reshape(-1, 2),
        )
        object.__setattr__(
            self,
            "depth",
            np.asarray(self.depth, dtype=np.float32).reshape(-1),
        )
        object.__setattr__(
            self,
            "source_indices",
            np.asarray(self.source_indices, dtype=np.int64).reshape(-1),
        )
        object.__setattr__(
            self,
            "camera_points",
            np.asarray(self.camera_points, dtype=np.float32).reshape(-1, 3),
        )


def camera_matrix_from_info(camera_info: Any) -> np.ndarray:
    """Extract a pinhole matrix from ROS CameraInfo, dict, or an array."""

    if isinstance(camera_info, np.ndarray):
        values = camera_info
    elif isinstance(camera_info, dict):
        values = camera_info.get("k", camera_info.get("K"))
    else:
        values = getattr(camera_info, "k", None)
        if values is None:
            values = getattr(camera_info, "K", None)
    if values is None:
        raise ValueError("camera_info does not provide a K matrix")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError("camera intrinsic matrix must contain nine values")
    matrix = matrix.reshape(3, 3)
    if (
        not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise ValueError("camera intrinsic matrix has invalid focal lengths")
    return matrix


def image_shape_from_info(camera_info: Any) -> tuple[int, int]:
    if isinstance(camera_info, dict):
        height = camera_info.get("height")
        width = camera_info.get("width")
    else:
        height = getattr(camera_info, "height", None)
        width = getattr(camera_info, "width", None)
    if height is None or width is None:
        raise ValueError("camera_info does not provide image dimensions")
    result = (int(height), int(width))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("image dimensions must be positive")
    return result


def transform_points(
    points: np.ndarray,
    tracking_to_camera: np.ndarray | None,
) -> np.ndarray:
    """Apply a caller-supplied TF transform into the optical camera frame."""

    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if tracking_to_camera is None:
        return source.astype(np.float32)
    transform = np.asarray(tracking_to_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("tracking_to_camera must be a finite 4x4 matrix")
    homogeneous = np.concatenate(
        (source, np.ones((len(source), 1), dtype=np.float64)),
        axis=1,
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def project_points(
    points: np.ndarray,
    camera_info: Any,
    image_shape: tuple[int, int] | None = None,
    *,
    tracking_to_camera: np.ndarray | None = None,
    minimum_depth: float = 0.02,
    maximum_depth: float | None = None,
) -> ProjectedPoints:
    """Project finite positive-z optical-frame points into image pixels."""

    matrix = camera_matrix_from_info(camera_info)
    if image_shape is None:
        image_shape = image_shape_from_info(camera_info)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")

    camera_points = transform_points(points, tracking_to_camera)
    source_indices = np.arange(len(camera_points), dtype=np.int64)
    valid = np.isfinite(camera_points).all(axis=1)
    valid &= camera_points[:, 2] > float(minimum_depth)
    if maximum_depth is not None:
        valid &= camera_points[:, 2] <= float(maximum_depth)
    camera_points = camera_points[valid]
    source_indices = source_indices[valid]
    if len(camera_points) == 0:
        return ProjectedPoints(
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
        )

    depth = camera_points[:, 2].astype(np.float64)
    u = matrix[0, 0] * camera_points[:, 0] / depth + matrix[0, 2]
    v = matrix[1, 1] * camera_points[:, 1] / depth + matrix[1, 2]
    uv = np.column_stack((u, v))
    inside = np.isfinite(uv).all(axis=1)
    inside &= (uv[:, 0] >= 0.0) & (uv[:, 0] < width)
    inside &= (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
    return ProjectedPoints(
        uv[inside].astype(np.float32),
        depth[inside].astype(np.float32),
        source_indices[inside],
        camera_points[inside],
    )


def rasterize_projection(
    uv: np.ndarray,
    image_shape: tuple[int, int],
    dilation_pixels: int = 0,
) -> np.ndarray:
    height, width = (int(image_shape[0]), int(image_shape[1]))
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels = np.rint(np.asarray(uv, dtype=np.float32)).astype(np.int64)
    if len(pixels):
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        pixels = pixels[valid]
        mask[pixels[:, 1], pixels[:, 0]] = 1
    dilation_pixels = max(int(dilation_pixels), 0)
    if dilation_pixels:
        size = dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def dense_positive_points(
    uv: np.ndarray,
    projection_mask: np.ndarray,
    *,
    maximum_points: int = 6,
    minimum_spacing_pixels: float = 8.0,
    boundary_margin_pixels: float = 1.0,
    density_blur_pixels: int = 9,
) -> np.ndarray:
    """Choose dense, interior projected pixels rather than a box center."""

    mask = np.asarray(projection_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("projection_mask must be 2D")
    height, width = mask.shape
    rounded = np.rint(np.asarray(uv, dtype=np.float32)).astype(np.int64)
    if len(rounded) == 0:
        return np.empty((0, 2), dtype=np.float32)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    rounded = rounded[valid]
    if len(rounded) == 0:
        return np.empty((0, 2), dtype=np.float32)
    rounded = np.unique(rounded, axis=0)

    density = np.zeros(mask.shape, dtype=np.float32)
    np.add.at(density, (rounded[:, 1], rounded[:, 0]), 1.0)
    blur = max(int(density_blur_pixels), 1)
    if blur % 2 == 0:
        blur += 1
    if blur > 1:
        density = cv2.GaussianBlur(density, (blur, blur), 0)
    interior_distance = cv2.distanceTransform(
        mask.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    scores = density[rounded[:, 1], rounded[:, 0]]
    scores *= 1.0 + interior_distance[rounded[:, 1], rounded[:, 0]]
    eligible = (
        interior_distance[rounded[:, 1], rounded[:, 0]]
        >= max(float(boundary_margin_pixels), 0.0)
    )
    if eligible.any():
        rounded = rounded[eligible]
        scores = scores[eligible]
    order = np.argsort(-scores, kind="stable")
    selected: list[np.ndarray] = []
    spacing = max(float(minimum_spacing_pixels), 0.0)
    for index in order:
        point = rounded[index].astype(np.float32)
        if selected and min(
            float(np.linalg.norm(point - existing))
            for existing in selected
        ) < spacing:
            continue
        selected.append(point)
        if len(selected) >= max(int(maximum_points), 1):
            break
    if not selected:
        selected.append(rounded[order[0]].astype(np.float32))
    return np.asarray(selected, dtype=np.float32)


def _padded_box(
    uv: np.ndarray,
    image_shape: tuple[int, int],
    config: ProjectionConfig,
) -> np.ndarray:
    height, width = image_shape
    minimum = np.min(uv, axis=0)
    maximum = np.max(uv, axis=0)
    size = np.maximum(maximum - minimum, 1.0)
    padding = (
        max(config.box_padding_pixels, 0)
        + max(config.box_padding_ratio, 0.0) * float(np.max(size))
    )
    box = np.array(
        [
            minimum[0] - padding,
            minimum[1] - padding,
            maximum[0] + padding,
            maximum[1] + padding,
        ],
        dtype=np.float32,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, max(width - 1, 0))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, max(height - 1, 0))
    return box


def _negative_points(
    box: np.ndarray,
    mask: np.ndarray,
    maximum_points: int,
) -> np.ndarray | None:
    if maximum_points <= 0:
        return None
    height, width = mask.shape
    x1, y1, x2, y2 = box
    candidates = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            [(x1 + x2) * 0.5, y1],
            [(x1 + x2) * 0.5, y2],
            [x1, (y1 + y2) * 0.5],
            [x2, (y1 + y2) * 0.5],
        ],
        dtype=np.float32,
    )
    selected: list[np.ndarray] = []
    for point in candidates:
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        if not mask[y, x]:
            selected.append(np.array([x, y], dtype=np.float32))
        if len(selected) >= maximum_points:
            break
    return None if not selected else np.asarray(selected, dtype=np.float32)


def points_to_prompt(
    points: np.ndarray,
    camera_info: Any,
    image_shape: tuple[int, int] | None,
    *,
    track_id: int,
    frame_index: int,
    tracking_to_camera: np.ndarray | None = None,
    config: ProjectionConfig | None = None,
    re_prompt: bool = False,
    reason: str = "",
) -> ProjectionPrompt | None:
    config = config or ProjectionConfig()
    if image_shape is None:
        image_shape = image_shape_from_info(camera_info)
    projection = project_points(
        points,
        camera_info,
        image_shape,
        tracking_to_camera=tracking_to_camera,
        minimum_depth=config.minimum_depth,
        maximum_depth=config.maximum_depth,
    )
    if len(projection.uv) == 0:
        return None
    mask = rasterize_projection(
        projection.uv,
        image_shape,
        config.projection_dilation_pixels,
    )
    positives = dense_positive_points(
        projection.uv,
        mask,
        maximum_points=config.maximum_positive_points,
        minimum_spacing_pixels=config.minimum_positive_spacing_pixels,
        boundary_margin_pixels=config.positive_boundary_margin_pixels,
        density_blur_pixels=config.density_blur_pixels,
    )
    if len(positives) == 0:
        return None
    box = _padded_box(projection.uv, image_shape, config)
    negatives = (
        _negative_points(box, mask, config.maximum_negative_points)
        if config.generate_negative_points
        else None
    )
    return ProjectionPrompt(
        track_id=track_id,
        frame_index=frame_index,
        box_xyxy=box,
        positive_points=positives,
        negative_points=negatives,
        projection_mask=mask,
        re_prompt=re_prompt,
        reason=reason,
    )


def project_cluster(
    cluster: Cluster3D,
    camera_info: Any,
    image_shape: tuple[int, int] | None = None,
    *,
    track_id: int | None = None,
    frame_index: int = 0,
    tracking_to_camera: np.ndarray | None = None,
    config: ProjectionConfig | None = None,
    re_prompt: bool = False,
    reason: str = "",
) -> ProjectionPrompt | None:
    return points_to_prompt(
        cluster.points,
        camera_info,
        image_shape,
        track_id=cluster.cluster_id if track_id is None else track_id,
        frame_index=frame_index,
        tracking_to_camera=tracking_to_camera,
        config=config,
        re_prompt=re_prompt,
        reason=reason,
    )


def project_track(
    track: TrackEstimate,
    camera_info: Any,
    image_shape: tuple[int, int] | None = None,
    *,
    frame_index: int = 0,
    tracking_to_camera: np.ndarray | None = None,
    config: ProjectionConfig | None = None,
    re_prompt: bool = False,
    reason: str = "",
) -> ProjectionPrompt | None:
    return points_to_prompt(
        track.source_points,
        camera_info,
        image_shape,
        track_id=track.track_id,
        frame_index=frame_index,
        tracking_to_camera=tracking_to_camera,
        config=config,
        re_prompt=re_prompt,
        reason=reason,
    )
