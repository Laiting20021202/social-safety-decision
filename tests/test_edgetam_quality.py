from __future__ import annotations

import numpy as np

from realtime_safety.edgetam_tracker.models import (
    AABB,
    Cluster3D,
    MaskQuality,
    OBB,
    PointCloudQuality,
)
from realtime_safety.edgetam_tracker.quality import (
    ConfidenceConfig,
    MaskQualityResult,
    MaskQualityConfig,
    PointCloudQualityConfig,
    RepromptConfig,
    compute_fused_confidence,
    decide_reprompt,
    evaluate_mask_quality,
    evaluate_pointcloud_quality,
)


def _cluster(
    count: int,
    *,
    missing_ratio: float = 0.0,
    quality_score: float = 1.0,
) -> Cluster3D:
    rng = np.random.default_rng(count)
    points = np.column_stack(
        (
            rng.uniform(-0.1, 0.1, count),
            rng.normal(1.0, 0.01, count),
            rng.uniform(-0.1, 0.1, count),
        )
    ).astype(np.float32)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    distances = np.linalg.norm(points, axis=1)
    nearest = int(np.argmin(distances))
    return Cluster3D(
        cluster_id=1,
        points=points,
        centroid=np.mean(points, axis=0),
        median_center=np.median(points, axis=0),
        aabb=AABB(minimum, maximum),
        obb=OBB(
            (minimum + maximum) * 0.5,
            maximum - minimum,
            np.eye(3, dtype=np.float32),
        ),
        nearest_point=points[nearest],
        nearest_distance=float(distances[nearest]),
        point_count=count,
        density=count / max(float(np.prod(maximum - minimum)), 1e-6),
        depth_variance=float(np.var(points[:, 1])),
        missing_depth_ratio=missing_ratio,
        quality_score=quality_score,
    )


def test_pointcloud_quality_distinguishes_good_sparse_and_invalid() -> None:
    good = evaluate_pointcloud_quality(_cluster(80))
    sparse = evaluate_pointcloud_quality(_cluster(6))
    invalid = evaluate_pointcloud_quality(_cluster(80, missing_ratio=0.99))
    assert good.quality is PointCloudQuality.GOOD
    assert good.score > sparse.score
    assert sparse.quality is PointCloudQuality.SPARSE
    assert invalid.quality is PointCloudQuality.INVALID
    assert "missing_depth" in invalid.reasons


def test_pointcloud_quality_fallback_uses_configured_depth_axis() -> None:
    cluster = _cluster(80)
    cluster.depth_variance = float("nan")
    result = evaluate_pointcloud_quality(
        cluster,
        config=PointCloudQualityConfig(depth_axis=2),
    )

    np.testing.assert_allclose(
        result.components["raw_depth_variance"],
        np.var(cluster.points[:, 2]),
    )


def test_mask_quality_combines_projection_depth_temporal_and_3d_consistency() -> None:
    projection = np.zeros((40, 40), dtype=bool)
    projection[10:30, 12:28] = True
    valid_depth = projection.copy()
    previous = projection.copy()
    result = evaluate_mask_quality(
        projection,
        projection,
        valid_depth_mask=valid_depth,
        previous_mask=previous,
        predicted_centroid=np.array([0.0, 1.0, 0.0]),
        measured_centroid=np.array([0.01, 1.0, 0.0]),
        model_score=0.95,
    )
    assert result.quality is MaskQuality.GOOD
    assert result.score > 0.9
    assert result.components["mask_cluster_iou"] == 1.0

    drifted = np.zeros_like(projection)
    drifted[0:8, 0:8] = True
    bad = evaluate_mask_quality(
        drifted,
        projection,
        valid_depth_mask=valid_depth,
        previous_mask=previous,
        predicted_centroid=np.array([0.0, 1.0, 0.0]),
        measured_centroid=np.array([2.0, 1.0, 0.0]),
        model_score=0.9,
    )
    assert bad.quality is MaskQuality.INVALID
    assert "projection_coverage" in bad.reasons
    assert "centroid_prediction" in bad.reasons


def test_sparse_pointcloud_depth_support_does_not_penalize_dense_mask_area() -> None:
    projection = np.zeros((40, 40), dtype=bool)
    projection[12, 12] = True
    projection[12, 26] = True
    projection[26, 12] = True
    projection[26, 26] = True
    valid_depth = projection.copy()
    mask = np.zeros_like(projection)
    mask[8:32, 8:32] = True

    result = evaluate_mask_quality(
        mask,
        projection,
        valid_depth_mask=valid_depth,
        config=MaskQualityConfig(
            minimum_valid_depth_points=4,
            minimum_valid_depth_ratio=0.8,
            degraded_score_threshold=0.25,
        ),
    )

    assert result.quality is not MaskQuality.INVALID
    assert result.components["valid_depth_points"] == 4.0
    assert result.components["projected_depth_points"] == 4.0
    assert result.components["valid_depth_ratio"] == 1.0


def test_sparse_mask_requires_an_absolute_minimum_depth_support_count() -> None:
    projection = np.zeros((20, 20), dtype=bool)
    projection[5, 5] = True
    projection[10, 10] = True
    projection[15, 15] = True
    result = evaluate_mask_quality(
        np.ones_like(projection),
        projection,
        valid_depth_mask=projection,
        config=MaskQualityConfig(
            minimum_valid_depth_points=4,
            minimum_valid_depth_ratio=0.0,
            degraded_score_threshold=0.0,
        ),
    )

    assert result.quality is MaskQuality.INVALID
    assert "insufficient_valid_depth_points" in result.reasons


def test_reprompt_reports_all_actionable_failures_and_prompt_order() -> None:
    result = MaskQualityResult(
        quality=MaskQuality.INVALID,
        score=0.1,
        components={
            "mask_cluster_iou": 0.05,
            "valid_depth_ratio": 0.1,
            "area_ratio": 3.0,
            "centroid_error": 0.8,
        },
        reasons=("projection_coverage",),
    )
    decision = decide_reprompt(
        result,
        frames_without_cluster_points=4,
        reappeared_after_occlusion=True,
        config=RepromptConfig(
            maximum_frames_without_cluster_points=2,
        ),
    )
    assert decision.required
    assert set(decision.reasons) >= {
        "mask_invalid",
        "low_mask_cluster_iou",
        "low_valid_depth",
        "mask_area_changed",
        "centroid_prediction_mismatch",
        "cluster_points_missing_from_mask",
        "track_reappeared",
    }
    assert decision.prompt_order == (
        "box_points",
        "projection_mask",
        "box",
    )


def test_unavailable_mask_redistributes_confidence_and_weak_evidence_adds_margin() -> None:
    config = ConfidenceConfig(
        pointcloud_weight=0.5,
        temporal_tracking_weight=0.3,
        mask_consistency_weight=0.2,
        unavailable_mask_redistribute=True,
        near_obstacle_distance=0.5,
        near_obstacle_confidence_floor=0.25,
    )
    without_mask = compute_fused_confidence(
        0.8,
        0.6,
        None,
        nearest_distance=1.0,
        config=config,
    )
    np.testing.assert_allclose(
        without_mask.raw_score,
        (0.5 * 0.8 + 0.3 * 0.6) / 0.8,
    )

    weak_nearby = compute_fused_confidence(
        0.05,
        0.05,
        0.0,
        nearest_distance=0.2,
        config=config,
    )
    assert weak_nearby.score == 0.25
    assert weak_nearby.uncertainty_margin > without_mask.uncertainty_margin
