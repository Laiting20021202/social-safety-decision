from __future__ import annotations

import numpy as np
import pytest

from realtime_safety.edgetam_tracker.models import (
    AABB,
    Cluster3D,
    OBB,
    PointCloudQuality,
    TrackingState,
)
from realtime_safety.edgetam_tracker.pointcloud_tracker import (
    PointCloudTracker,
    PointCloudTrackerConfig,
)


def _cluster(
    cluster_id: int,
    center: tuple[float, float, float],
    *,
    size: tuple[float, float, float] = (0.2, 0.2, 0.2),
    point_count: int = 27,
    nearest_distance: float | None = None,
) -> Cluster3D:
    center_array = np.asarray(center, dtype=np.float32)
    size_array = np.asarray(size, dtype=np.float32)
    rng = np.random.default_rng(cluster_id + 100)
    points = center_array + rng.uniform(
        -size_array * 0.5,
        size_array * 0.5,
        size=(point_count, 3),
    ).astype(np.float32)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    distances = np.linalg.norm(points, axis=1)
    nearest_index = int(np.argmin(distances))
    return Cluster3D(
        cluster_id=cluster_id,
        points=points,
        centroid=np.mean(points, axis=0),
        median_center=np.median(points, axis=0),
        aabb=AABB(minimum, maximum),
        obb=OBB(
            (minimum + maximum) * 0.5,
            maximum - minimum,
            np.eye(3, dtype=np.float32),
        ),
        nearest_point=points[nearest_index],
        nearest_distance=(
            float(distances[nearest_index])
            if nearest_distance is None
            else nearest_distance
        ),
        point_count=point_count,
        source_indices=np.arange(point_count),
        density=point_count / max(float(np.prod(maximum - minimum)), 1e-6),
        depth_variance=float(np.var(points[:, 1])),
        quality=PointCloudQuality.GOOD,
        quality_score=0.9,
    )


def test_tracker_confirms_and_keeps_ids_with_size_and_count_cues() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=2,
            emergency_confirmation_distance=0.0,
            maximum_association_distance=0.8,
            maximum_mahalanobis_distance=20.0,
            measured_velocity_blend=0.6,
        )
    )
    first = tracker.update(
        [
            _cluster(10, (-0.45, 1.0, 0.0), size=(0.12, 0.12, 0.12), point_count=20),
            _cluster(20, (0.45, 1.0, 0.0), size=(0.35, 0.35, 0.35), point_count=80),
        ],
        0.0,
    )
    assert [track.state for track in first] == [
        TrackingState.TENTATIVE,
        TrackingState.TENTATIVE,
    ]

    second = tracker.update(
        [
            _cluster(11, (-0.20, 1.0, 0.0), size=(0.12, 0.12, 0.12), point_count=20),
            _cluster(21, (0.20, 1.0, 0.0), size=(0.35, 0.35, 0.35), point_count=80),
        ],
        0.2,
    )
    assert [track.track_id for track in second] == [1, 2]
    assert all(track.state is TrackingState.CONFIRMED for track in second)

    crossed = tracker.update(
        [
            _cluster(12, (0.08, 1.0, 0.0), size=(0.12, 0.12, 0.12), point_count=20),
            _cluster(22, (-0.08, 1.0, 0.0), size=(0.35, 0.35, 0.35), point_count=80),
        ],
        0.4,
    )
    by_id = {track.track_id: track for track in crossed}
    assert by_id[1].point_count == 20
    assert by_id[2].point_count == 80
    assert by_id[1].velocity[0] > 0.0
    assert by_id[2].velocity[0] < 0.0
    assert len(by_id[1].predicted_positions) == 15
    assert by_id[1].predicted_positions[-1, 0] > by_id[1].position[0]


def test_optional_mask_iou_cost_disambiguates_equal_candidates() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=1,
            emergency_confirmation_distance=0.0,
            maximum_association_distance=1.0,
            maximum_mahalanobis_distance=20.0,
            centroid_cost_weight=0.1,
            aabb_size_cost_weight=0.0,
            point_count_cost_weight=0.0,
            mask_iou_cost_weight=0.9,
        )
    )
    tracker.update(
        [
            _cluster(100, (-0.1, 1.0, 0.0)),
            _cluster(200, (0.1, 1.0, 0.0)),
        ],
        0.0,
    )
    result = tracker.update(
        [
            _cluster(300, (0.0, 1.0, 0.0), point_count=31),
            _cluster(400, (0.0, 1.0, 0.0), point_count=47),
        ],
        0.1,
        mask_ious={
            (1, 300): 0.05,
            (1, 400): 0.95,
            (2, 300): 0.95,
            (2, 400): 0.05,
        },
    )
    by_id = {track.track_id: track for track in result}
    assert by_id[1].point_count == 47
    assert by_id[2].point_count == 31


def test_confirmed_track_transitions_occluded_lost_deleted_then_is_purged() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=2,
            emergency_confirmation_distance=0.0,
            maximum_occluded_frames=2,
            occluded_retention_seconds=0.3,
            lost_retention_seconds=1.0,
            maximum_missed_frames=10,
            maximum_prediction_age_seconds=0.5,
        )
    )
    tracker.update([_cluster(1, (0.8, 1.0, 0.0))], 0.0)
    confirmed = tracker.update([_cluster(2, (0.85, 1.0, 0.0))], 0.1)
    assert confirmed[0].state is TrackingState.CONFIRMED

    occluded = tracker.update([], 0.2)
    assert occluded[0].state is TrackingState.OCCLUDED
    assert occluded[0].source_points.size > 0
    assert occluded[0].pointcloud_quality is PointCloudQuality.INVALID

    lost = tracker.update([], 0.5)
    assert lost[0].state is TrackingState.LOST
    assert lost[0].uncertainty_margin > confirmed[0].uncertainty_margin

    deleted = tracker.update([], 1.2)
    assert deleted[0].state is TrackingState.DELETED
    assert tracker.update([], 1.3) == []


def test_near_tentative_cluster_is_emergency_promoted_and_not_suppressed() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=5,
            emergency_confirmation_distance=0.4,
            emergency_confidence_floor=0.35,
        )
    )
    result = tracker.update(
        [
            _cluster(
                1,
                (0.25, 0.0, 0.0),
                nearest_distance=0.20,
            )
        ],
        0.0,
    )
    assert result[0].state is TrackingState.CONFIRMED
    assert result[0].hit_count == 1
    assert result[0].confidence >= 0.35


def test_explicit_prediction_horizons_are_exactly_configurable() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=1,
            emergency_confirmation_distance=0.0,
            prediction_horizons_seconds=(1.0, 0.2, 0.5, 0.5),
        )
    )
    estimate = tracker.update(
        [_cluster(1, (0.4, 1.0, 0.0))], 0.0
    )[0]
    assert estimate.predicted_positions.shape == (3, 3)
    expected = (
        estimate.position[None, :]
        + np.array([[0.2], [0.5], [1.0]], dtype=np.float32)
        * estimate.velocity[None, :]
    )
    np.testing.assert_allclose(estimate.predicted_positions, expected)


def test_dynamic_robot_origin_updates_estimated_and_predicted_distance() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=1,
            emergency_confirmation_distance=0.0,
            maximum_prediction_age_seconds=1.0,
            measured_velocity_blend=1.0,
        )
    )
    tracker.update([_cluster(1, (0.2, 1.0, 0.0))], 0.0)
    tracker.update([_cluster(2, (0.3, 1.0, 0.0))], 0.1)

    new_origin = np.array((0.5, 1.0, 0.0), dtype=np.float32)
    tracker.set_robot_origin(new_origin)
    current = tracker.predict_to(0.1)[0]
    current_distances = np.linalg.norm(
        current.source_points - new_origin,
        axis=1,
    )
    assert current.nearest_distance == pytest.approx(
        float(np.min(current_distances))
    )
    np.testing.assert_allclose(
        current.nearest_point,
        current.source_points[int(np.argmin(current_distances))],
    )

    predicted = tracker.predict_to(0.25)[0]
    predicted_distances = np.linalg.norm(
        predicted.source_points - new_origin,
        axis=1,
    )
    assert predicted.nearest_distance == pytest.approx(
        float(np.min(predicted_distances))
    )
    assert tuple(tracker.config.robot_origin) == pytest.approx(tuple(new_origin))

    returned_origin = tracker.robot_origin
    returned_origin[0] = 99.0
    np.testing.assert_allclose(tracker.robot_origin, new_origin)

    for invalid in (
        (0.0, 0.0),
        (0.0, np.nan, 0.0),
        (0.0, 0.0, np.inf),
    ):
        with pytest.raises(ValueError, match="exactly three finite"):
            tracker.set_robot_origin(invalid)
        np.testing.assert_allclose(tracker.robot_origin, new_origin)


def test_measured_surface_is_never_shifted_by_kalman_centroid_lag() -> None:
    tracker = PointCloudTracker(
        PointCloudTrackerConfig(
            confirmation_hits=1,
            emergency_confirmation_distance=0.0,
            maximum_association_distance=2.0,
            maximum_mahalanobis_distance=100.0,
            initial_position_variance=1e-6,
            measurement_variance=100.0,
            measured_velocity_blend=0.0,
        )
    )
    tracker.update([_cluster(10, (0.0, 1.0, 0.0))], 0.0)
    nearer_measurement = _cluster(11, (0.0, 0.30, 0.0))
    estimate = tracker.update([nearer_measurement], 0.1)[0]

    # The deliberately sluggish filter centroid remains far away, while the
    # safety cloud and clearance exactly follow the new raw measurement.
    assert estimate.position[1] > nearer_measurement.centroid[1] + 0.5
    np.testing.assert_allclose(
        estimate.source_points,
        nearer_measurement.points,
    )
    np.testing.assert_allclose(
        estimate.aabb.minimum,
        nearer_measurement.aabb.minimum,
    )
    np.testing.assert_allclose(
        estimate.aabb.maximum,
        nearer_measurement.aabb.maximum,
    )
    measured_distances = np.linalg.norm(
        nearer_measurement.points,
        axis=1,
    )
    assert estimate.nearest_distance == pytest.approx(
        float(np.min(measured_distances))
    )
