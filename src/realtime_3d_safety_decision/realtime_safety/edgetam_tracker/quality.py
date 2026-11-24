from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from realtime_safety.edgetam_tracker.models import (
    AABB,
    Cluster3D,
    MaskObservation,
    MaskQuality,
    PointCloudQuality,
)


@dataclass(slots=True)
class PointCloudQualityConfig:
    """Thresholds used to grade a geometric obstacle observation."""

    depth_axis: int = 1
    minimum_valid_points: int = 3
    minimum_good_points: int = 30
    target_density: float = 300.0
    maximum_good_depth_variance: float = 0.04
    maximum_invalid_depth_variance: float = 0.25
    maximum_good_missing_depth_ratio: float = 0.35
    maximum_invalid_missing_depth_ratio: float = 0.95
    minimum_physical_size: float = 0.005
    maximum_physical_size: float = 5.0
    innovation_scale: float = 0.5
    good_score_threshold: float = 0.68
    point_count_weight: float = 0.24
    density_weight: float = 0.18
    depth_variance_weight: float = 0.15
    missing_depth_weight: float = 0.12
    physical_size_weight: float = 0.10
    size_consistency_weight: float = 0.10
    innovation_weight: float = 0.11

    def __post_init__(self) -> None:
        if self.depth_axis not in {0, 1, 2}:
            raise ValueError("depth_axis must be 0, 1, or 2")


@dataclass(frozen=True, slots=True)
class PointCloudQualityResult:
    quality: PointCloudQuality
    score: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class MaskQualityConfig:
    """Quality gates for deciding whether a mask may refine geometry."""

    minimum_mask_pixels: int = 12
    good_score_threshold: float = 0.70
    degraded_score_threshold: float = 0.40
    minimum_good_projection_coverage: float = 0.60
    minimum_valid_projection_coverage: float = 0.10
    minimum_good_valid_depth_ratio: float = 0.55
    minimum_valid_depth_ratio: float = 0.08
    minimum_valid_depth_points: int = 3
    maximum_good_centroid_error: float = 0.25
    maximum_valid_centroid_error: float = 1.0
    model_score_weight: float = 0.12
    projection_coverage_weight: float = 0.21
    mask_cluster_iou_weight: float = 0.20
    valid_depth_weight: float = 0.17
    temporal_iou_weight: float = 0.12
    prediction_consistency_weight: float = 0.10
    area_consistency_weight: float = 0.08


@dataclass(frozen=True, slots=True)
class MaskQualityResult:
    quality: MaskQuality
    score: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    mask_present: bool = True


@dataclass(slots=True)
class RepromptConfig:
    minimum_mask_cluster_iou: float = 0.25
    minimum_valid_depth_ratio: float = 0.25
    minimum_area_ratio: float = 0.45
    maximum_area_ratio: float = 2.2
    maximum_centroid_error: float = 0.5
    maximum_frames_without_cluster_points: int = 2


@dataclass(frozen=True, slots=True)
class RepromptDecision:
    required: bool
    reasons: tuple[str, ...] = ()
    prompt_order: tuple[str, ...] = ("box_points", "projection_mask", "box")


@dataclass(slots=True)
class ConfidenceConfig:
    pointcloud_weight: float = 0.45
    temporal_tracking_weight: float = 0.35
    mask_consistency_weight: float = 0.20
    unavailable_mask_redistribute: bool = True
    near_obstacle_distance: float = 0.5
    near_obstacle_confidence_floor: float = 0.20
    base_uncertainty_margin: float = 0.02
    low_confidence_uncertainty_gain: float = 0.35


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    score: float
    raw_score: float
    uncertainty_margin: float
    components: dict[str, float] = field(default_factory=dict)


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _weighted_average(values: list[tuple[float, float]]) -> float:
    usable = [(float(value), max(float(weight), 0.0)) for value, weight in values]
    total = sum(weight for _, weight in usable)
    if total <= 1e-12:
        return 0.0
    return _clamp01(sum(value * weight for value, weight in usable) / total)


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("masks must have matching shapes")
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return 0.0 if union == 0 else intersection / union


def evaluate_pointcloud_quality(
    cluster: Cluster3D,
    previous_aabb: AABB | None = None,
    config: PointCloudQualityConfig | None = None,
) -> PointCloudQualityResult:
    """Grade a cluster without treating a sparse cluster as absent."""

    config = config or PointCloudQualityConfig()
    points = np.asarray(cluster.points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    finite_count = int(np.count_nonzero(finite))
    reasons: list[str] = []

    if finite_count < config.minimum_valid_points:
        return PointCloudQualityResult(
            PointCloudQuality.INVALID,
            0.0,
            {"finite_points": float(finite_count)},
            ("too_few_finite_points",),
        )

    size = np.asarray(cluster.aabb.size, dtype=np.float64)
    if not np.isfinite(size).all():
        return PointCloudQualityResult(
            PointCloudQuality.INVALID,
            0.0,
            {"finite_points": float(finite_count)},
            ("nonfinite_geometry",),
        )

    count_span = max(
        config.minimum_good_points - config.minimum_valid_points,
        1,
    )
    point_count_score = _clamp01(
        (finite_count - config.minimum_valid_points) / count_span
    )

    volume = max(float(cluster.aabb.volume), 1e-6)
    density = float(cluster.density)
    if not np.isfinite(density) or density <= 0.0:
        density = finite_count / volume
    density_score = _clamp01(density / max(config.target_density, 1e-6))

    depth_variance = float(cluster.depth_variance)
    if not np.isfinite(depth_variance) or depth_variance < 0.0:
        depth_variance = float(
            np.var(points[finite, int(config.depth_axis)])
        )
    variance_span = max(
        config.maximum_invalid_depth_variance
        - config.maximum_good_depth_variance,
        1e-9,
    )
    depth_variance_score = _clamp01(
        1.0
        - max(
            depth_variance - config.maximum_good_depth_variance,
            0.0,
        )
        / variance_span
    )

    missing_ratio = _clamp01(cluster.missing_depth_ratio)
    missing_depth_score = 1.0 - missing_ratio

    dimensions_valid = bool(
        np.all(size <= config.maximum_physical_size)
        and np.any(size >= config.minimum_physical_size)
    )
    physical_size_score = 1.0 if dimensions_valid else 0.0

    size_consistency_score = 1.0
    if previous_aabb is not None:
        previous_size = np.maximum(
            np.asarray(previous_aabb.size, dtype=np.float64),
            config.minimum_physical_size,
        )
        current_size = np.maximum(size, config.minimum_physical_size)
        log_difference = float(
            np.mean(np.abs(np.log(current_size / previous_size)))
        )
        size_consistency_score = float(np.exp(-log_difference))

    innovation = max(float(cluster.innovation_distance), 0.0)
    innovation_score = float(
        np.exp(-innovation / max(config.innovation_scale, 1e-6))
    )

    components = {
        "finite_points": float(finite_count),
        "point_count": point_count_score,
        "density": density_score,
        "depth_variance": depth_variance_score,
        "missing_depth": missing_depth_score,
        "physical_size": physical_size_score,
        "size_consistency": size_consistency_score,
        "innovation": innovation_score,
        "raw_density": density,
        "raw_depth_variance": depth_variance,
        "raw_missing_depth_ratio": missing_ratio,
    }
    score = _weighted_average(
        [
            (point_count_score, config.point_count_weight),
            (density_score, config.density_weight),
            (depth_variance_score, config.depth_variance_weight),
            (missing_depth_score, config.missing_depth_weight),
            (physical_size_score, config.physical_size_weight),
            (size_consistency_score, config.size_consistency_weight),
            (innovation_score, config.innovation_weight),
        ]
    )

    hard_invalid = False
    if missing_ratio >= config.maximum_invalid_missing_depth_ratio:
        reasons.append("missing_depth")
        hard_invalid = True
    if depth_variance >= config.maximum_invalid_depth_variance:
        reasons.append("depth_variance")
        hard_invalid = True
    if not dimensions_valid:
        reasons.append("physical_size")
        hard_invalid = True

    if hard_invalid:
        quality = PointCloudQuality.INVALID
    elif (
        finite_count >= config.minimum_good_points
        and missing_ratio <= config.maximum_good_missing_depth_ratio
        and score >= config.good_score_threshold
    ):
        quality = PointCloudQuality.GOOD
    else:
        quality = PointCloudQuality.SPARSE
        reasons.append("below_good_threshold")
    return PointCloudQualityResult(quality, score, components, tuple(reasons))


def apply_pointcloud_quality(
    cluster: Cluster3D,
    previous_aabb: AABB | None = None,
    config: PointCloudQualityConfig | None = None,
) -> PointCloudQualityResult:
    result = evaluate_pointcloud_quality(cluster, previous_aabb, config)
    cluster.quality = result.quality
    cluster.quality_score = result.score
    return result


def evaluate_mask_quality(
    mask: np.ndarray | None,
    projection_mask: np.ndarray,
    *,
    valid_depth_mask: np.ndarray | None = None,
    previous_mask: np.ndarray | None = None,
    predicted_centroid: np.ndarray | None = None,
    measured_centroid: np.ndarray | None = None,
    model_score: float | None = None,
    config: MaskQualityConfig | None = None,
) -> MaskQualityResult:
    """Score temporal, 2D, depth, and 3D consistency of an EdgeTAM mask."""

    config = config or MaskQualityConfig()
    projection = np.asarray(projection_mask, dtype=bool)
    if projection.ndim != 2:
        raise ValueError("projection_mask must be a 2D array")
    if mask is None:
        return MaskQualityResult(
            MaskQuality.UNAVAILABLE,
            0.0,
            {},
            ("mask_missing",),
            mask_present=False,
        )

    candidate = np.asarray(mask, dtype=bool)
    if candidate.shape != projection.shape:
        return MaskQualityResult(
            MaskQuality.INVALID,
            0.0,
            {},
            ("shape_mismatch",),
        )

    mask_pixels = int(np.count_nonzero(candidate))
    projection_pixels = int(np.count_nonzero(projection))
    if mask_pixels < config.minimum_mask_pixels:
        return MaskQualityResult(
            MaskQuality.INVALID,
            0.0,
            {"mask_pixels": float(mask_pixels)},
            ("mask_too_small",),
        )

    intersection = int(np.count_nonzero(candidate & projection))
    projection_coverage = (
        0.0 if projection_pixels == 0 else intersection / projection_pixels
    )
    mask_cluster_iou = _binary_iou(candidate, projection)
    components: dict[str, float] = {
        "mask_pixels": float(mask_pixels),
        "projection_pixels": float(projection_pixels),
        "projection_coverage": projection_coverage,
        "mask_cluster_iou": mask_cluster_iou,
    }
    weighted: list[tuple[float, float]] = [
        (projection_coverage, config.projection_coverage_weight),
        (mask_cluster_iou, config.mask_cluster_iou_weight),
    ]

    if model_score is not None and np.isfinite(model_score):
        components["model_score"] = _clamp01(model_score)
        weighted.append((components["model_score"], config.model_score_weight))

    valid_depth_ratio: float | None = None
    if valid_depth_mask is not None:
        valid_depth = np.asarray(valid_depth_mask, dtype=bool)
        if valid_depth.shape != candidate.shape:
            raise ValueError("valid_depth_mask must match mask shape")
        # PointCloud2 is commonly voxel-downsampled and therefore much sparser
        # than the RGB mask.  Dividing valid samples by the semantic mask area
        # makes every otherwise-correct mask fail merely because most RGB
        # pixels have no retained point.  Instead, measure whether the mask
        # contains the depth samples supporting its projected 3D proposal.
        valid_depth_points = int(
            np.count_nonzero(candidate & valid_depth)
        )
        projected_depth_points = int(
            np.count_nonzero(projection & valid_depth)
        )
        valid_depth_ratio = _clamp01(
            valid_depth_points / max(projected_depth_points, 1)
        )
        components["valid_depth_points"] = float(valid_depth_points)
        components["projected_depth_points"] = float(
            projected_depth_points
        )
        components["valid_depth_ratio"] = valid_depth_ratio
        weighted.append((valid_depth_ratio, config.valid_depth_weight))

    if previous_mask is not None:
        previous = np.asarray(previous_mask, dtype=bool)
        if previous.shape != candidate.shape:
            raise ValueError("previous_mask must match mask shape")
        temporal_iou = _binary_iou(candidate, previous)
        previous_area = int(np.count_nonzero(previous))
        area_ratio = (
            float("inf") if previous_area == 0 else mask_pixels / previous_area
        )
        area_consistency = (
            0.0
            if not np.isfinite(area_ratio) or area_ratio <= 0.0
            else min(area_ratio, 1.0 / area_ratio)
        )
        components["temporal_iou"] = temporal_iou
        components["area_ratio"] = area_ratio
        components["area_consistency"] = area_consistency
        weighted.extend(
            [
                (temporal_iou, config.temporal_iou_weight),
                (area_consistency, config.area_consistency_weight),
            ]
        )

    centroid_error: float | None = None
    if predicted_centroid is not None and measured_centroid is not None:
        predicted = np.asarray(predicted_centroid, dtype=np.float64).reshape(3)
        measured = np.asarray(measured_centroid, dtype=np.float64).reshape(3)
        centroid_error = float(np.linalg.norm(predicted - measured))
        consistency = float(
            np.exp(
                -centroid_error
                / max(config.maximum_good_centroid_error, 1e-6)
            )
        )
        components["centroid_error"] = centroid_error
        components["prediction_consistency"] = consistency
        weighted.append(
            (consistency, config.prediction_consistency_weight)
        )

    score = _weighted_average(weighted)
    reasons: list[str] = []
    hard_invalid = projection_pixels == 0
    if projection_pixels == 0:
        reasons.append("projection_missing")
    if projection_coverage < config.minimum_valid_projection_coverage:
        reasons.append("projection_coverage")
        hard_invalid = True
    if (
        valid_depth_ratio is not None
        and valid_depth_ratio < config.minimum_valid_depth_ratio
    ):
        reasons.append("valid_depth")
        hard_invalid = True
    if (
        valid_depth_mask is not None
        and components.get("valid_depth_points", 0.0)
        < max(int(config.minimum_valid_depth_points), 1)
    ):
        reasons.append("insufficient_valid_depth_points")
        hard_invalid = True
    if (
        centroid_error is not None
        and centroid_error > config.maximum_valid_centroid_error
    ):
        reasons.append("centroid_prediction")
        hard_invalid = True

    good_critical_values = (
        projection_coverage >= config.minimum_good_projection_coverage
        and (
            valid_depth_ratio is None
            or valid_depth_ratio >= config.minimum_good_valid_depth_ratio
        )
        and (
            centroid_error is None
            or centroid_error <= config.maximum_good_centroid_error
        )
    )
    if hard_invalid or score < config.degraded_score_threshold:
        quality = MaskQuality.INVALID
    elif score >= config.good_score_threshold and good_critical_values:
        quality = MaskQuality.GOOD
    else:
        quality = MaskQuality.DEGRADED
        reasons.append("below_good_threshold")
    return MaskQualityResult(quality, score, components, tuple(reasons))


def apply_mask_quality(
    observation: MaskObservation,
    projection_mask: np.ndarray,
    **kwargs: object,
) -> MaskQualityResult:
    result = evaluate_mask_quality(
        observation.mask,
        projection_mask,
        model_score=observation.model_score,
        **kwargs,
    )
    observation.quality = result.quality
    observation.quality_score = result.score
    observation.components = dict(result.components)
    return result


def decide_reprompt(
    result: MaskQualityResult | None,
    *,
    cluster_present: bool = True,
    previous_mask_present: bool = True,
    frames_without_cluster_points: int = 0,
    reappeared_after_occlusion: bool = False,
    config: RepromptConfig | None = None,
) -> RepromptDecision:
    """Return all deterministic re-prompt reasons in priority order."""

    config = config or RepromptConfig()
    if not cluster_present:
        return RepromptDecision(False, ("cluster_unavailable",))

    reasons: list[str] = []
    if result is None or not result.mask_present:
        if previous_mask_present:
            reasons.append("mask_disappeared")
        else:
            reasons.append("mask_missing")
    else:
        if result.quality is MaskQuality.INVALID:
            reasons.append("mask_invalid")
        iou = result.components.get("mask_cluster_iou")
        if iou is not None and iou < config.minimum_mask_cluster_iou:
            reasons.append("low_mask_cluster_iou")
        valid_depth = result.components.get("valid_depth_ratio")
        if (
            valid_depth is not None
            and valid_depth < config.minimum_valid_depth_ratio
        ):
            reasons.append("low_valid_depth")
        area_ratio = result.components.get("area_ratio")
        if area_ratio is not None and (
            not np.isfinite(area_ratio)
            or area_ratio < config.minimum_area_ratio
            or area_ratio > config.maximum_area_ratio
        ):
            reasons.append("mask_area_changed")
        centroid_error = result.components.get("centroid_error")
        if (
            centroid_error is not None
            and centroid_error > config.maximum_centroid_error
        ):
            reasons.append("centroid_prediction_mismatch")

    if (
        frames_without_cluster_points
        > config.maximum_frames_without_cluster_points
    ):
        reasons.append("cluster_points_missing_from_mask")
    if reappeared_after_occlusion:
        reasons.append("track_reappeared")
    return RepromptDecision(bool(reasons), tuple(dict.fromkeys(reasons)))


def compute_fused_confidence(
    pointcloud_score: float,
    temporal_tracking_score: float,
    mask_consistency_score: float | None,
    *,
    nearest_distance: float | None = None,
    config: ConfidenceConfig | None = None,
) -> ConfidenceResult:
    """Combine sources while increasing margin when evidence is weak."""

    config = config or ConfidenceConfig()
    pointcloud = _clamp01(pointcloud_score)
    temporal = _clamp01(temporal_tracking_score)
    mask = (
        None
        if mask_consistency_score is None
        else _clamp01(mask_consistency_score)
    )
    weighted = [
        (pointcloud, config.pointcloud_weight),
        (temporal, config.temporal_tracking_weight),
    ]
    if mask is not None or not config.unavailable_mask_redistribute:
        weighted.append(
            (
                0.0 if mask is None else mask,
                config.mask_consistency_weight,
            )
        )
    raw_score = _weighted_average(weighted)
    score = raw_score
    if (
        nearest_distance is not None
        and nearest_distance <= config.near_obstacle_distance
    ):
        score = max(score, config.near_obstacle_confidence_floor)
    uncertainty_margin = (
        config.base_uncertainty_margin
        + config.low_confidence_uncertainty_gain * (1.0 - raw_score)
    )
    components = {
        "pointcloud": pointcloud,
        "temporal_tracking": temporal,
        "mask_consistency": 0.0 if mask is None else mask,
        "mask_available": float(mask is not None),
    }
    return ConfidenceResult(
        _clamp01(score),
        raw_score,
        float(max(uncertainty_margin, 0.0)),
        components,
    )
