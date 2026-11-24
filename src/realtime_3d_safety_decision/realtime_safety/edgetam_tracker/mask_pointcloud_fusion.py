from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from realtime_safety.edgetam_tracker.models import (
    AABB,
    CloudFrame,
    Cluster3D,
    MaskObservation,
    MaskQuality,
    OBB,
    TrackEstimate,
)


@dataclass(slots=True)
class FusionConfig:
    morphology_kernel_size: int = 3
    erosion_iterations: int = 1
    dilation_iterations: int = 0
    minimum_component_pixels: int = 12
    maximum_components: int = 1
    allow_mask_resize: bool = False
    aabb_gate_margin: float = 0.08
    depth_axis: int = 1
    absolute_depth_gate: float = 0.15
    relative_depth_gate: float = 0.12
    depth_mad_scale: float = 4.0
    center_gate_margin: float = 0.10
    minimum_center_gate_radius: float = 0.08
    robust_distance_mad_scale: float = 5.0
    minimum_fused_points: int = 6
    robot_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.morphology_kernel_size < 1:
            raise ValueError("morphology_kernel_size must be positive")
        if self.minimum_fused_points < 1:
            raise ValueError("minimum_fused_points must be positive")
        if self.depth_axis not in (0, 1, 2):
            raise ValueError("depth_axis must be 0, 1, or 2")
        if self.maximum_components < 1:
            raise ValueError("maximum_components must be positive")


@dataclass(slots=True)
class FusionResult:
    points: np.ndarray
    colors: np.ndarray | None
    source_indices: np.ndarray | None
    centroid: np.ndarray
    median_center: np.ndarray
    aabb: AABB
    obb: OBB
    nearest_point: np.ndarray
    nearest_distance: float
    processed_mask: np.ndarray
    used_mask: bool
    used_fallback: bool
    reason: str
    mask_candidate_points: int
    fused_point_count: int

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        if self.source_indices is not None:
            self.source_indices = np.asarray(
                self.source_indices,
                dtype=np.int64,
            ).reshape(-1)
        self.centroid = np.asarray(self.centroid, dtype=np.float32).reshape(3)
        self.median_center = np.asarray(
            self.median_center,
            dtype=np.float32,
        ).reshape(3)
        self.nearest_point = np.asarray(
            self.nearest_point,
            dtype=np.float32,
        ).reshape(3)
        self.processed_mask = np.asarray(self.processed_mask, dtype=bool)


def clean_mask(
    mask: np.ndarray,
    *,
    projection_mask: np.ndarray | None = None,
    config: FusionConfig | None = None,
) -> np.ndarray:
    """Apply configurable morphology and connected-component selection."""

    config = config or FusionConfig()
    cleaned = np.asarray(mask, dtype=bool)
    if cleaned.ndim != 2:
        raise ValueError("mask must be a 2D array")
    binary = cleaned.astype(np.uint8)
    kernel_size = max(int(config.morphology_kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    if config.erosion_iterations > 0:
        binary = cv2.erode(
            binary,
            kernel,
            iterations=int(config.erosion_iterations),
        )
    if config.dilation_iterations > 0:
        binary = cv2.dilate(
            binary,
            kernel,
            iterations=int(config.dilation_iterations),
        )
    if not np.any(binary):
        return binary.astype(bool)

    projection: np.ndarray | None = None
    if projection_mask is not None:
        projection = np.asarray(projection_mask, dtype=bool)
        if projection.shape != binary.shape:
            raise ValueError("projection_mask must match mask shape")

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    candidates: list[tuple[int, int, int]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.minimum_component_pixels:
            continue
        overlap = (
            0
            if projection is None
            else int(np.count_nonzero((labels == label) & projection))
        )
        candidates.append((overlap, area, label))
    if not candidates:
        return np.zeros_like(binary, dtype=bool)

    if projection is not None and max(item[0] for item in candidates) > 0:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        candidates.sort(key=lambda item: item[1], reverse=True)
    kept = {
        item[2]
        for item in candidates[: max(int(config.maximum_components), 1)]
    }
    return np.isin(labels, list(kept))


def fuse_mask_with_cloud(
    mask: np.ndarray | MaskObservation | None,
    high_resolution_cloud: CloudFrame,
    cluster: Cluster3D,
    *,
    predicted_geometry: AABB | TrackEstimate | None = None,
    projection_mask: np.ndarray | None = None,
    mask_quality: MaskQuality | object | None = None,
    config: FusionConfig | None = None,
    point_depths: np.ndarray | None = None,
    cluster_depths: np.ndarray | None = None,
) -> FusionResult:
    """Refine with mask/depth while retaining point-cloud geometry on failure.

    ``point_depths`` and ``cluster_depths`` should be optical-camera z values
    when the cloud has already been transformed into a world/base tracking
    frame. If omitted, the pure-core fallback uses ``config.depth_axis``.
    """

    config = config or FusionConfig()
    observation: MaskObservation | None = None
    if isinstance(mask, MaskObservation):
        observation = mask
        mask_array: np.ndarray | None = observation.mask
        if mask_quality is None:
            mask_quality = observation.quality
    else:
        mask_array = mask
    resolved_quality = getattr(mask_quality, "quality", mask_quality)

    fallback_mask_shape = (
        high_resolution_cloud.image_shape
        if high_resolution_cloud.image_shape is not None
        else (
            np.asarray(mask_array).shape
            if mask_array is not None and np.asarray(mask_array).ndim == 2
            else (0, 0)
        )
    )
    empty_mask = np.zeros(fallback_mask_shape, dtype=bool)
    if mask_array is None:
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            empty_mask,
            config,
            "mask_unavailable",
        )
    if resolved_quality in (MaskQuality.INVALID, MaskQuality.UNAVAILABLE):
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            np.asarray(mask_array, dtype=bool),
            config,
            f"mask_quality_{resolved_quality.value.lower()}",
        )

    candidate_mask = np.asarray(mask_array, dtype=bool)
    image_shape = _cloud_image_shape(high_resolution_cloud)
    if candidate_mask.shape != image_shape:
        if not config.allow_mask_resize:
            return _fallback_result(
                cluster,
                high_resolution_cloud,
                candidate_mask,
                config,
                "mask_shape_mismatch",
            )
        candidate_mask = cv2.resize(
            candidate_mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if projection_mask is not None:
            projection_mask = cv2.resize(
                np.asarray(projection_mask, dtype=np.uint8),
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    processed_mask = clean_mask(
        candidate_mask,
        projection_mask=projection_mask,
        config=config,
    )
    if not np.any(processed_mask):
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            processed_mask,
            config,
            "mask_empty_after_morphology",
        )

    pixels = _cloud_pixels(high_resolution_cloud, image_shape)
    if pixels is None:
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            processed_mask,
            config,
            "cloud_pixel_correspondence_unavailable",
        )
    width = image_shape[1]
    height = image_shape[0]
    valid_pixel = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    selected = np.zeros(len(high_resolution_cloud.points), dtype=bool)
    selected[valid_pixel] = processed_mask[
        pixels[valid_pixel, 1],
        pixels[valid_pixel, 0],
    ]
    finite = np.isfinite(high_resolution_cloud.points).all(axis=1)
    selected &= finite
    mask_candidate_points = int(np.count_nonzero(selected))
    if mask_candidate_points < config.minimum_fused_points:
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            processed_mask,
            config,
            "insufficient_mask_points",
            mask_candidate_points,
        )

    gate = selected.copy()
    predicted_aabb: AABB | None
    predicted_center: np.ndarray
    if isinstance(predicted_geometry, TrackEstimate):
        predicted_aabb = predicted_geometry.aabb
        predicted_center = predicted_geometry.position
    elif isinstance(predicted_geometry, AABB):
        predicted_aabb = predicted_geometry
        predicted_center = predicted_geometry.center
    else:
        predicted_aabb = cluster.aabb
        predicted_center = cluster.median_center

    if predicted_aabb is not None:
        expanded = predicted_aabb.expanded(max(config.aabb_gate_margin, 0.0))
        points = high_resolution_cloud.points
        gate &= np.all(points >= expanded.minimum, axis=1)
        gate &= np.all(points <= expanded.maximum, axis=1)

    cluster_points = np.asarray(cluster.points, dtype=np.float32)
    cluster_points = cluster_points[np.isfinite(cluster_points).all(axis=1)]
    if len(cluster_points):
        if point_depths is None:
            cloud_depth_values = high_resolution_cloud.points[
                :, config.depth_axis
            ]
        else:
            cloud_depth_values = np.asarray(
                point_depths, dtype=np.float32
            ).reshape(-1)
            if len(cloud_depth_values) != len(
                high_resolution_cloud.points
            ):
                raise ValueError(
                    "point_depths must match high_resolution_cloud"
                )
        if cluster_depths is None:
            reference_depths = cluster_points[:, config.depth_axis]
        else:
            reference_depths = np.asarray(
                cluster_depths, dtype=np.float32
            ).reshape(-1)
            if len(reference_depths) != len(cluster.points):
                raise ValueError("cluster_depths must match cluster.points")
            reference_depths = reference_depths[
                np.isfinite(cluster.points).all(axis=1)
            ]
        reference_depths = reference_depths[
            np.isfinite(reference_depths)
        ]
        if len(reference_depths) == 0:
            return _fallback_result(
                cluster,
                high_resolution_cloud,
                processed_mask,
                config,
                "camera_depth_unavailable",
                mask_candidate_points,
            )
        reference_depth = float(np.median(reference_depths))
        mad = float(
            np.median(np.abs(reference_depths - reference_depth))
        )
        depth_threshold = max(
            config.absolute_depth_gate,
            config.relative_depth_gate * max(abs(reference_depth), 1e-3),
            config.depth_mad_scale * 1.4826 * mad,
        )
        gate &= np.isfinite(cloud_depth_values)
        gate &= (
            np.abs(cloud_depth_values - reference_depth)
            <= depth_threshold
        )

    half_diagonal = 0.5 * float(np.linalg.norm(cluster.aabb.size))
    center_radius = max(
        half_diagonal + config.center_gate_margin,
        config.minimum_center_gate_radius,
    )
    gate &= (
        np.linalg.norm(
            high_resolution_cloud.points - predicted_center,
            axis=1,
        )
        <= center_radius
    )

    selected_indices = np.flatnonzero(gate)
    if len(selected_indices) < config.minimum_fused_points:
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            processed_mask,
            config,
            "insufficient_gated_points",
            mask_candidate_points,
        )

    selected_indices = _robust_indices(
        high_resolution_cloud.points,
        selected_indices,
        config.robust_distance_mad_scale,
    )
    if len(selected_indices) < config.minimum_fused_points:
        return _fallback_result(
            cluster,
            high_resolution_cloud,
            processed_mask,
            config,
            "insufficient_robust_points",
            mask_candidate_points,
        )

    mask_points = high_resolution_cloud.points[selected_indices].astype(
        np.float32,
        copy=True,
    )
    mask_colors = _select_optional(
        high_resolution_cloud.colors,
        selected_indices,
        len(high_resolution_cloud.points),
    )
    mask_source_indices = high_resolution_cloud.source_indices[
        selected_indices
    ]
    points, colors, source_indices = _conservative_union(
        cluster,
        mask_points,
        mask_colors,
        mask_source_indices,
    )
    return _geometry_result(
        points,
        colors,
        source_indices,
        processed_mask,
        config,
        used_mask=True,
        used_fallback=False,
        reason="mask_depth_fused_conservative_union",
        mask_candidate_points=mask_candidate_points,
    )


def _cloud_image_shape(cloud: CloudFrame) -> tuple[int, int]:
    if cloud.image_shape is not None:
        height, width = cloud.image_shape
        if height > 0 and width > 0:
            return int(height), int(width)
    if cloud.pixels_uv is not None and len(cloud.pixels_uv):
        width = int(np.max(cloud.pixels_uv[:, 0])) + 1
        height = int(np.max(cloud.pixels_uv[:, 1])) + 1
        if height > 0 and width > 0:
            return height, width
    raise ValueError("high-resolution cloud has no image shape")


def _cloud_pixels(
    cloud: CloudFrame,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if cloud.pixels_uv is not None and len(cloud.pixels_uv) == len(cloud.points):
        return cloud.pixels_uv
    height, width = image_shape
    if len(cloud.points) == height * width:
        y, x = np.indices((height, width))
        return np.column_stack((x.reshape(-1), y.reshape(-1))).astype(
            np.int32
        )
    return None


def _robust_indices(
    points: np.ndarray,
    indices: np.ndarray,
    mad_scale: float,
) -> np.ndarray:
    selected = points[indices]
    if len(selected) < 8 or mad_scale <= 0.0:
        return indices
    median = np.median(selected, axis=0)
    distances = np.linalg.norm(selected - median, axis=1)
    distance_median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - distance_median)))
    threshold = distance_median + max(
        float(mad_scale) * 1.4826 * mad,
        1e-3,
    )
    return indices[distances <= threshold]


def _select_optional(
    values: np.ndarray | None,
    indices: np.ndarray,
    expected_count: int,
) -> np.ndarray | None:
    if values is None or len(values) != expected_count:
        return None
    return np.asarray(values)[indices].copy()


def _conservative_union(
    cluster: Cluster3D,
    mask_points: np.ndarray,
    mask_colors: np.ndarray | None,
    mask_source_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Add mask/depth detail without removing measured safety surfaces.

    A segmentation mask can be partial even when its aggregate quality passes.
    Replacing the measured cluster with that subset could increase reported
    clearance. The safety geometry is therefore the deduplicated union of the
    current measured cluster and the gated high-resolution mask points.
    """

    base_points = np.asarray(cluster.points, dtype=np.float32).reshape(-1, 3)
    refined_points = np.asarray(mask_points, dtype=np.float32).reshape(-1, 3)
    combined_points = np.concatenate((base_points, refined_points), axis=0)

    base_indices = (
        None
        if cluster.source_indices is None
        else np.asarray(cluster.source_indices, dtype=np.int64).reshape(-1)
    )
    refined_indices = (
        None
        if mask_source_indices is None
        else np.asarray(mask_source_indices, dtype=np.int64).reshape(-1)
    )
    use_source_ids = (
        base_indices is not None
        and refined_indices is not None
        and len(base_indices) == len(base_points)
        and len(refined_indices) == len(refined_points)
    )
    if use_source_ids:
        combined_indices = np.concatenate(
            (base_indices, refined_indices),
            axis=0,
        )
        _, first = np.unique(combined_indices, return_index=True)
        keep = np.sort(first)
        source_indices: np.ndarray | None = combined_indices[keep]
    else:
        _, first = np.unique(combined_points, axis=0, return_index=True)
        keep = np.sort(first)
        source_indices = None

    base_colors = (
        None
        if cluster.colors is None
        else np.asarray(cluster.colors, dtype=np.uint8).reshape(-1, 3)
    )
    refined_colors = (
        None
        if mask_colors is None
        else np.asarray(mask_colors, dtype=np.uint8).reshape(-1, 3)
    )
    if (
        base_colors is not None
        and refined_colors is not None
        and len(base_colors) == len(base_points)
        and len(refined_colors) == len(refined_points)
    ):
        colors: np.ndarray | None = np.concatenate(
            (base_colors, refined_colors),
            axis=0,
        )[keep]
    else:
        colors = None
    return combined_points[keep], colors, source_indices


def _fallback_result(
    cluster: Cluster3D,
    cloud: CloudFrame,
    processed_mask: np.ndarray,
    config: FusionConfig,
    reason: str,
    mask_candidate_points: int = 0,
) -> FusionResult:
    # ``Cluster3D.source_indices`` are stable sensor/source IDs, not guaranteed
    # to be positional indices into this CloudFrame after filtering. The
    # cluster already contains the exact high-resolution fallback geometry.
    points = cluster.points.astype(np.float32, copy=True)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = (
        None
        if cluster.colors is None or len(cluster.colors) != len(finite)
        else cluster.colors[finite].copy()
    )
    source_indices = (
        None
        if cluster.source_indices is None
        or len(cluster.source_indices) != len(finite)
        else cluster.source_indices[finite].copy()
    )
    if len(points) == 0:
        # ClusterExtractor should never produce this, but fail loudly instead
        # of publishing an empty obstacle as a valid fallback.
        raise ValueError("point-cloud fallback contains no finite points")
    return _geometry_result(
        points,
        colors,
        source_indices,
        processed_mask,
        config,
        used_mask=False,
        used_fallback=True,
        reason=reason,
        mask_candidate_points=mask_candidate_points,
    )


def _geometry_result(
    points: np.ndarray,
    colors: np.ndarray | None,
    source_indices: np.ndarray | None,
    processed_mask: np.ndarray,
    config: FusionConfig,
    *,
    used_mask: bool,
    used_fallback: bool,
    reason: str,
    mask_candidate_points: int,
) -> FusionResult:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    centroid = np.mean(points, axis=0).astype(np.float32)
    median_center = np.median(points, axis=0).astype(np.float32)
    aabb = AABB(np.min(points, axis=0), np.max(points, axis=0))
    obb = _oriented_box(points)
    origin = np.asarray(config.robot_origin, dtype=np.float32)
    distances = np.linalg.norm(points - origin, axis=1)
    nearest_index = int(np.argmin(distances))
    return FusionResult(
        points=points,
        colors=colors,
        source_indices=source_indices,
        centroid=centroid,
        median_center=median_center,
        aabb=aabb,
        obb=obb,
        nearest_point=points[nearest_index],
        nearest_distance=float(distances[nearest_index]),
        processed_mask=processed_mask,
        used_mask=used_mask,
        used_fallback=used_fallback,
        reason=reason,
        mask_candidate_points=int(mask_candidate_points),
        fused_point_count=len(points),
    )


def _oriented_box(points: np.ndarray) -> OBB:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.mean(points, axis=0)
    if len(points) < 3 or np.allclose(points, points[0]):
        rotation = np.eye(3, dtype=np.float32)
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        return OBB(
            ((minimum + maximum) * 0.5).astype(np.float32),
            (maximum - minimum).astype(np.float32),
            rotation,
        )
    covariance = np.cov(points - center, rowvar=False)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        rotation = np.eye(3, dtype=np.float64)
    else:
        _, vectors = np.linalg.eigh(covariance)
        rotation = vectors[:, ::-1]
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
    local = (points - center) @ rotation
    minimum = np.min(local, axis=0)
    maximum = np.max(local, axis=0)
    local_center = (minimum + maximum) * 0.5
    world_center = center + local_center @ rotation.T
    return OBB(
        world_center.astype(np.float32),
        (maximum - minimum).astype(np.float32),
        rotation.astype(np.float32),
    )
