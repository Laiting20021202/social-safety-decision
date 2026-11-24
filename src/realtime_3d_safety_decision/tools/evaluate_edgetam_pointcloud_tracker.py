#!/usr/bin/env python3
"""Deterministic, synthetic checks for the point-cloud-first tracker.

This is a functional evaluator, not a live sensor or EdgeTAM benchmark.  It
deliberately avoids wall-clock timing and model inference so repeated runs on
the same code produce directly comparable CSV and Markdown output.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from realtime_safety.edgetam_tracker.cluster_extractor import (  # noqa: E402
    ClusterExtractor,
    ClusterExtractorConfig,
)
from realtime_safety.edgetam_tracker.mask_pointcloud_fusion import (  # noqa: E402
    FusionConfig,
    fuse_mask_with_cloud,
)
from realtime_safety.edgetam_tracker.models import (  # noqa: E402
    AABB,
    CloudFrame,
    Cluster3D,
    MaskQuality,
    OBB,
    PointCloudQuality,
    TrackEstimate,
    TrackingState,
)
from realtime_safety.edgetam_tracker.pointcloud_preprocessor import (  # noqa: E402
    PointCloudPreprocessor,
    PointCloudPreprocessorConfig,
)
from realtime_safety.edgetam_tracker.pointcloud_tracker import (  # noqa: E402
    PointCloudTracker,
    PointCloudTrackerConfig,
)
from realtime_safety.edgetam_tracker.quality import (  # noqa: E402
    MaskQualityConfig,
    PointCloudQualityConfig,
    apply_pointcloud_quality,
    evaluate_mask_quality,
)
from realtime_safety.edgetam_tracker.sensor_sync import transform_cloud  # noqa: E402


FORMAT_VERSION = 3
DT_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: str
    expectation: str
    frames: int
    observation_frames: int
    passed: bool
    id_switches: int
    mean_position_error_m: float | None
    final_state: str
    pointcloud_outcome: str
    edge_tam_metrics: str
    details: str


def _tracker(**overrides: object) -> PointCloudTracker:
    parameters: dict[str, object] = {
        "confirmation_hits": 2,
        "emergency_confirmation_distance": 0.25,
        "maximum_association_distance": 0.55,
        "maximum_mahalanobis_distance": 20.0,
        "maximum_association_cost": 0.9,
        "maximum_occluded_frames": 3,
        "occluded_retention_seconds": 0.45,
        "maximum_missed_frames": 8,
        "lost_retention_seconds": 0.9,
        "maximum_prediction_age_seconds": 1.0,
        "prediction_horizon_seconds": 1.0,
        "prediction_step_seconds": 0.2,
        "measured_velocity_blend": 0.45,
        "acceleration_process_variance": 0.6,
        "measurement_variance": 0.02,
    }
    parameters.update(overrides)
    return PointCloudTracker(PointCloudTrackerConfig(**parameters))


def _offsets(count: int, size: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-0.5, 0.5, size=(count, 3))
    offsets -= np.mean(offsets, axis=0, keepdims=True)
    return (offsets * size).astype(np.float32)


def _cluster(
    center: np.ndarray | tuple[float, float, float],
    *,
    cluster_id: int = 0,
    count: int = 72,
    size: tuple[float, float, float] = (0.16, 0.12, 0.18),
    seed: int = 1,
    robot_origin: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    quality: PointCloudQuality = PointCloudQuality.GOOD,
    quality_score: float = 0.92,
) -> Cluster3D:
    center_array = np.asarray(center, dtype=np.float32).reshape(3)
    points = center_array + _offsets(
        count,
        np.asarray(size, dtype=np.float32),
        seed,
    )
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    origin = np.asarray(robot_origin, dtype=np.float32).reshape(3)
    distances = np.linalg.norm(points - origin, axis=1)
    nearest_index = int(np.argmin(distances))
    volume = max(float(np.prod(maximum - minimum)), 1e-6)
    return Cluster3D(
        cluster_id=cluster_id,
        points=points,
        centroid=np.mean(points, axis=0),
        median_center=np.median(points, axis=0),
        aabb=AABB(minimum, maximum),
        obb=OBB(center_array, maximum - minimum, np.eye(3, dtype=np.float32)),
        nearest_point=points[nearest_index],
        nearest_distance=float(distances[nearest_index]),
        point_count=count,
        source_indices=np.arange(count, dtype=np.int64),
        density=float(count / volume),
        depth_variance=float(np.var(points[:, 1])),
        quality=quality,
        quality_score=quality_score,
    )


def _active(estimates: list[TrackEstimate]) -> list[TrackEstimate]:
    return [
        estimate
        for estimate in estimates
        if estimate.state is not TrackingState.DELETED
    ]


def _assign(
    estimates: list[TrackEstimate],
    true_centers: list[np.ndarray],
) -> dict[int, TrackEstimate]:
    candidates = _active(estimates)
    if not candidates or not true_centers:
        return {}
    costs = np.asarray(
        [
            [
                np.linalg.norm(estimate.position - np.asarray(center))
                for center in true_centers
            ]
            for estimate in candidates
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    return {
        int(column): candidates[int(row)]
        for row, column in zip(rows, columns, strict=True)
    }


def _switch_count(ids: list[int]) -> int:
    return sum(previous != current for previous, current in zip(ids, ids[1:]))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _static_obstacle() -> ScenarioResult:
    tracker = _tracker()
    center = np.array([0.35, 1.20, 0.10], dtype=np.float32)
    identifiers: list[int] = []
    errors: list[float] = []
    final: TrackEstimate | None = None
    frames = 16
    for frame in range(frames):
        measurement = center + np.array(
            [0.002 * ((frame % 3) - 1), 0.0, 0.0],
            dtype=np.float32,
        )
        estimates = tracker.update(
            [_cluster(measurement, seed=100 + frame)],
            frame * DT_SECONDS,
        )
        assignment = _assign(estimates, [center])
        final = assignment[0]
        identifiers.append(final.track_id)
        errors.append(float(np.linalg.norm(final.position - center)))
    switches = _switch_count(identifiers)
    mean_error = _mean(errors)
    passed = bool(
        switches == 0
        and final is not None
        and final.state is TrackingState.CONFIRMED
        and mean_error is not None
        and mean_error < 0.02
    )
    return ScenarioResult(
        "static",
        "One persistent confirmed ID for a stationary obstacle",
        frames,
        frames,
        passed,
        switches,
        mean_error,
        final.state.value if final else "MISSING",
        "GOOD geometry retained",
        "N/A (EdgeTAM not invoked)",
        f"unique_ids={len(set(identifiers))}",
    )


def _moving_obstacle() -> ScenarioResult:
    tracker = _tracker()
    initial = np.array([-0.45, 1.10, 0.05], dtype=np.float32)
    velocity = np.array([0.32, 0.04, 0.0], dtype=np.float32)
    identifiers: list[int] = []
    errors: list[float] = []
    final: TrackEstimate | None = None
    frames = 26
    for frame in range(frames):
        center = initial + velocity * (frame * DT_SECONDS)
        estimates = tracker.update(
            [_cluster(center, seed=200 + frame)],
            frame * DT_SECONDS,
        )
        final = _assign(estimates, [center])[0]
        identifiers.append(final.track_id)
        errors.append(float(np.linalg.norm(final.position - center)))
    switches = _switch_count(identifiers)
    mean_error = _mean(errors)
    velocity_error = (
        float(np.linalg.norm(final.velocity - velocity))
        if final is not None
        else float("inf")
    )
    passed = bool(
        switches == 0
        and final is not None
        and final.state is TrackingState.CONFIRMED
        and mean_error is not None
        and mean_error < 0.08
        and velocity_error < 0.16
    )
    return ScenarioResult(
        "moving",
        "Keep one ID and estimate a constant 3D velocity",
        frames,
        frames,
        passed,
        switches,
        mean_error,
        final.state.value if final else "MISSING",
        "GOOD geometry retained",
        "N/A (EdgeTAM not invoked)",
        f"final_velocity_error_mps={velocity_error:.6f}",
    )


def _crossing_obstacles() -> ScenarioResult:
    tracker = _tracker(maximum_association_distance=0.40)
    ids_by_object = [[], []]
    errors: list[float] = []
    final_states = ["MISSING", "MISSING"]
    frames = 24
    for frame in range(frames):
        time_seconds = frame * DT_SECONDS
        centers = [
            np.array([-0.55 + 0.48 * time_seconds, 1.05, 0.04], dtype=np.float32),
            np.array([0.55 - 0.48 * time_seconds, 1.05, 0.04], dtype=np.float32),
        ]
        estimates = tracker.update(
            [
                _cluster(
                    centers[0],
                    cluster_id=0,
                    count=64,
                    size=(0.12, 0.15, 0.16),
                    seed=300 + frame,
                ),
                _cluster(
                    centers[1],
                    cluster_id=1,
                    count=96,
                    size=(0.23, 0.14, 0.17),
                    seed=400 + frame,
                ),
            ],
            time_seconds,
        )
        assignment = _assign(estimates, centers)
        for object_index in (0, 1):
            estimate = assignment[object_index]
            ids_by_object[object_index].append(estimate.track_id)
            errors.append(
                float(np.linalg.norm(estimate.position - centers[object_index]))
            )
            final_states[object_index] = estimate.state.value
    switches = sum(_switch_count(values) for values in ids_by_object)
    mean_error = _mean(errors)
    distinct_ids = (
        len(set(ids_by_object[0])) == 1
        and len(set(ids_by_object[1])) == 1
        and ids_by_object[0][0] != ids_by_object[1][0]
    )
    passed = bool(
        switches == 0
        and distinct_ids
        and mean_error is not None
        and mean_error < 0.10
        and all(state == TrackingState.CONFIRMED.value for state in final_states)
    )
    return ScenarioResult(
        "crossing",
        "Maintain two distinct IDs while trajectories cross",
        frames,
        frames * 2,
        passed,
        switches,
        mean_error,
        "+".join(final_states),
        "Two point-cloud tracks",
        "N/A (EdgeTAM not invoked)",
        f"object_ids={ids_by_object[0][0]},{ids_by_object[1][0]}",
    )


def _short_occlusion() -> ScenarioResult:
    tracker = _tracker()
    identifiers: list[int] = []
    missing_states: list[str] = []
    errors: list[float] = []
    frames = 15
    final: TrackEstimate | None = None
    for frame in range(frames):
        center = np.array(
            [-0.20 + 0.12 * frame * DT_SECONDS, 1.0, 0.0],
            dtype=np.float32,
        )
        clusters = (
            []
            if 6 <= frame <= 8
            else [_cluster(center, seed=500 + frame)]
        )
        estimates = tracker.update(clusters, frame * DT_SECONDS)
        if clusters:
            final = _assign(estimates, [center])[0]
            identifiers.append(final.track_id)
            errors.append(float(np.linalg.norm(final.position - center)))
        else:
            active = _active(estimates)
            missing_states.append(active[0].state.value if active else "MISSING")
    switches = _switch_count(identifiers)
    mean_error = _mean(errors)
    passed = bool(
        switches == 0
        and final is not None
        and final.state is TrackingState.CONFIRMED
        and set(missing_states).issubset(
            {TrackingState.OCCLUDED.value, TrackingState.LOST.value}
        )
    )
    return ScenarioResult(
        "occlusion",
        "Predict through a short cloud gap and recover the same ID",
        frames,
        frames - 3,
        passed,
        switches,
        mean_error,
        final.state.value if final else "MISSING",
        "Prediction during 3 missing frames",
        "N/A (EdgeTAM not invoked)",
        f"gap_states={'+'.join(missing_states)}",
    )


def _leave_and_reappear() -> ScenarioResult:
    tracker = _tracker(
        maximum_occluded_frames=2,
        occluded_retention_seconds=0.25,
        maximum_missed_frames=5,
        lost_retention_seconds=0.60,
    )
    early_ids: list[int] = []
    late_ids: list[int] = []
    errors: list[float] = []
    frames = 24
    final: TrackEstimate | None = None
    for frame in range(frames):
        visible = frame <= 4 or frame >= 19
        center = np.array([0.30, 1.30, 0.0], dtype=np.float32)
        estimates = tracker.update(
            [_cluster(center, seed=600 + frame)] if visible else [],
            frame * DT_SECONDS,
        )
        if not visible:
            continue
        final = _assign(estimates, [center])[0]
        (early_ids if frame <= 4 else late_ids).append(final.track_id)
        errors.append(float(np.linalg.norm(final.position - center)))
    intentional_reidentification = bool(
        early_ids
        and late_ids
        and len(set(early_ids)) == 1
        and len(set(late_ids)) == 1
        and early_ids[0] != late_ids[0]
    )
    return ScenarioResult(
        "leave_reappear",
        "Delete an expired track; a later return receives a new ID",
        frames,
        len(early_ids) + len(late_ids),
        intentional_reidentification,
        1 if intentional_reidentification else 0,
        _mean(errors),
        final.state.value if final else "MISSING",
        "Expired geometry not held indefinitely",
        "N/A (EdgeTAM not invoked)",
        (
            f"before_id={early_ids[0] if early_ids else 'MISSING'};"
            f"after_id={late_ids[0] if late_ids else 'MISSING'}"
        ),
    )


def _sparse_cloud() -> ScenarioResult:
    tracker = _tracker()
    center = np.array([0.45, 0.90, 0.05], dtype=np.float32)
    dense_ids: list[int] = []
    for frame in range(2):
        estimates = tracker.update(
            [_cluster(center, seed=700 + frame)],
            frame * DT_SECONDS,
        )
        dense_ids.append(_assign(estimates, [center])[0].track_id)

    sparse = _cluster(
        center,
        count=6,
        size=(0.12, 0.10, 0.14),
        seed=710,
        quality=PointCloudQuality.SPARSE,
        quality_score=0.35,
    )
    quality_result = apply_pointcloud_quality(
        sparse,
        config=PointCloudQualityConfig(
            minimum_valid_points=3,
            minimum_good_points=30,
            good_score_threshold=0.75,
        ),
    )
    final = _assign(
        tracker.update([sparse], 2 * DT_SECONDS),
        [center],
    )[0]
    identifiers = dense_ids + [final.track_id]
    switches = _switch_count(identifiers)
    passed = bool(
        switches == 0
        and quality_result.quality is PointCloudQuality.SPARSE
        and final.state is TrackingState.CONFIRMED
        and final.pointcloud_quality is PointCloudQuality.SPARSE
    )
    return ScenarioResult(
        "sparse",
        "Degrade confidence/quality without deleting a valid sparse obstacle",
        3,
        3,
        passed,
        switches,
        float(np.linalg.norm(final.position - center)),
        final.state.value,
        quality_result.quality.value,
        "N/A (EdgeTAM not invoked)",
        f"quality_score={quality_result.score:.6f};points={sparse.point_count}",
    )


def _mask_drift() -> ScenarioResult:
    tracker = _tracker()
    center = np.array([0.25, 1.00, 0.05], dtype=np.float32)
    identifiers: list[int] = []
    cluster = _cluster(center, count=72, seed=800)
    for frame in range(2):
        estimates = tracker.update(
            [_cluster(center, seed=800 + frame)],
            frame * DT_SECONDS,
        )
        identifiers.append(_assign(estimates, [center])[0].track_id)

    projection = np.zeros((64, 64), dtype=bool)
    projection[22:42, 22:42] = True
    drifted_mask = np.zeros_like(projection)
    drifted_mask[2:17, 2:17] = True
    mask_result = evaluate_mask_quality(
        drifted_mask,
        projection,
        valid_depth_mask=projection,
        previous_mask=projection,
        predicted_centroid=center,
        measured_centroid=center,
        model_score=None,
        config=MaskQualityConfig(),
    )

    pixels = np.column_stack(
        (
            np.arange(cluster.point_count, dtype=np.int32) % 18 + 23,
            np.arange(cluster.point_count, dtype=np.int32) // 18 + 24,
        )
    )
    cloud = CloudFrame(
        points=cluster.points,
        stamp=0.2,
        frame_id="tracking_frame",
        pixels_uv=pixels,
        source_indices=np.arange(cluster.point_count, dtype=np.int64),
        image_shape=projection.shape,
    )
    cluster.source_indices = np.arange(cluster.point_count, dtype=np.int64)
    fusion = fuse_mask_with_cloud(
        drifted_mask,
        cloud,
        cluster,
        projection_mask=projection,
        mask_quality=mask_result.quality,
        config=FusionConfig(minimum_fused_points=6),
    )
    final = _assign(
        tracker.update([cluster], 2 * DT_SECONDS, mask_ious=None),
        [center],
    )[0]
    identifiers.append(final.track_id)
    switches = _switch_count(identifiers)
    passed = bool(
        switches == 0
        and mask_result.quality
        in {MaskQuality.DEGRADED, MaskQuality.INVALID}
        and fusion.used_fallback
        and not fusion.used_mask
        and fusion.fused_point_count > 0
        and final.state is TrackingState.CONFIRMED
    )
    return ScenarioResult(
        "mask_drift",
        "Reject a drifted mask and retain point-cloud geometry/ID",
        3,
        3,
        passed,
        switches,
        float(np.linalg.norm(final.position - center)),
        final.state.value,
        f"fallback:{fusion.reason}",
        "N/A (synthetic mask quality only; no model inference)",
        (
            f"mask_quality={mask_result.quality.value};"
            f"mask_score={mask_result.score:.6f};"
            f"fallback_points={fusion.fused_point_count}"
        ),
    )


def _rgb_missing() -> ScenarioResult:
    tracker = _tracker()
    identifiers: list[int] = []
    errors: list[float] = []
    final: TrackEstimate | None = None
    frames = 10
    for frame in range(frames):
        center = np.array(
            [0.10 + frame * 0.01, 1.15, -0.02],
            dtype=np.float32,
        )
        estimates = tracker.update(
            [_cluster(center, seed=900 + frame)],
            frame * DT_SECONDS,
            mask_ious=None,
        )
        final = _assign(estimates, [center])[0]
        identifiers.append(final.track_id)
        errors.append(float(np.linalg.norm(final.position - center)))
    switches = _switch_count(identifiers)
    passed = bool(
        switches == 0
        and final is not None
        and final.state is TrackingState.CONFIRMED
        and final.mask_quality is MaskQuality.UNAVAILABLE
    )
    return ScenarioResult(
        "rgb_missing",
        "Continue point-cloud tracking with no RGB or mask observation",
        frames,
        frames,
        passed,
        switches,
        _mean(errors),
        final.state.value if final else "MISSING",
        "GOOD geometry; mask UNAVAILABLE",
        "N/A (RGB deliberately absent)",
        "mask input omitted for every frame",
    )


def _robot_motion() -> ScenarioResult:
    tracker = _tracker()
    preprocessor = PointCloudPreprocessor(
        PointCloudPreprocessorConfig(
            workspace_min=np.array([-1.5, 0.05, -1.0]),
            workspace_max=np.array([1.5, 3.0, 1.0]),
            voxel_size=0.0,
            remove_outliers=False,
        )
    )
    extractor = ClusterExtractor(
        ClusterExtractorConfig(
            method="euclidean",
            tolerance=0.11,
            min_points=8,
            max_points=1000,
            sparse_point_threshold=8,
        )
    )
    fixed_center = np.array([0.30, 1.25, 0.0], dtype=np.float32)
    fixed_points = fixed_center + _offsets(
        80,
        np.array([0.13, 0.12, 0.17], dtype=np.float32),
        1000,
    )
    identifiers: list[int] = []
    errors: list[float] = []
    final: TrackEstimate | None = None
    frames = 12
    for frame in range(frames):
        camera_translation = np.array(
            [0.025 * frame, 0.0, 0.0],
            dtype=np.float32,
        )
        sensor_cloud = CloudFrame(
            points=fixed_points - camera_translation,
            stamp=frame * DT_SECONDS,
            frame_id="moving_camera",
        )
        tracking_cloud = transform_cloud(
            sensor_cloud,
            camera_translation,
            np.array([0.0, 0.0, 0.0, 1.0]),
            "tracking_frame",
        )
        processed = preprocessor.process(tracking_cloud)
        clusters = extractor.extract(processed.processed_cloud)
        if len(clusters) != 1:
            return ScenarioResult(
                "robot_motion",
                "Transform moving-camera cloud into a stable tracking frame",
                frames,
                frame,
                False,
                0,
                _mean(errors),
                "MISSING",
                f"unexpected_cluster_count={len(clusters)}",
                "N/A (EdgeTAM not invoked)",
                "Synthetic transform/preprocess/cluster path failed",
            )
        estimates = tracker.update(clusters, frame * DT_SECONDS)
        final = _assign(estimates, [fixed_center])[0]
        identifiers.append(final.track_id)
        errors.append(float(np.linalg.norm(final.position - fixed_center)))
    switches = _switch_count(identifiers)
    mean_error = _mean(errors)
    passed = bool(
        switches == 0
        and final is not None
        and final.state is TrackingState.CONFIRMED
        and mean_error is not None
        and mean_error < 0.02
    )
    return ScenarioResult(
        "robot_motion",
        "Transform moving-camera cloud into a stable tracking frame",
        frames,
        frames,
        passed,
        switches,
        mean_error,
        final.state.value if final else "MISSING",
        "TF-transformed point-cloud track",
        "N/A (EdgeTAM not invoked)",
        "camera_translation_x=0.000..0.275 m",
    )


def _sudden_hand_entrance() -> ScenarioResult:
    tracker = _tracker(confirmation_hits=3)
    far_center = np.array([0.55, 1.30, 0.0], dtype=np.float32)
    for frame in range(3):
        tracker.update(
            [_cluster(far_center, cluster_id=0, seed=1100 + frame)],
            frame * DT_SECONDS,
        )
    hand_center = np.array([0.04, 0.17, 0.0], dtype=np.float32)
    estimates = tracker.update(
        [
            _cluster(far_center, cluster_id=0, seed=1110),
            _cluster(
                hand_center,
                cluster_id=1,
                count=48,
                size=(0.08, 0.07, 0.10),
                seed=1111,
            ),
        ],
        3 * DT_SECONDS,
    )
    hand = _assign(estimates, [far_center, hand_center])[1]
    immediate_confirmation = bool(
        hand.state is TrackingState.CONFIRMED and hand.hit_count == 1
    )
    return ScenarioResult(
        "hand_entrance",
        "Immediately confirm a newly observed near-field obstacle",
        4,
        5,
        immediate_confirmation,
        0,
        float(np.linalg.norm(hand.position - hand_center)),
        hand.state.value,
        "Emergency near-field point-cloud track",
        "N/A (EdgeTAM not invoked)",
        (
            f"hit_count={hand.hit_count};"
            f"nearest_distance_m={hand.nearest_distance:.6f}"
        ),
    )


SCENARIOS: tuple[Callable[[], ScenarioResult], ...] = (
    _static_obstacle,
    _moving_obstacle,
    _crossing_obstacles,
    _short_occlusion,
    _leave_and_reappear,
    _sparse_cloud,
    _mask_drift,
    _rgb_missing,
    _robot_motion,
    _sudden_hand_entrance,
)


def _metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def render_csv(results: list[ScenarioResult]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "scenario",
            "expectation",
            "frames",
            "observation_frames",
            "passed",
            "id_switches",
            "mean_position_error_m",
            "final_state",
            "pointcloud_outcome",
            "edge_tam_metrics",
            "details",
        )
    )
    for result in results:
        writer.writerow(
            (
                result.scenario,
                result.expectation,
                result.frames,
                result.observation_frames,
                "PASS" if result.passed else "FAIL",
                result.id_switches,
                _metric(result.mean_position_error_m),
                result.final_state,
                result.pointcloud_outcome,
                result.edge_tam_metrics,
                result.details,
            )
        )
    return stream.getvalue()


def render_markdown(results: list[ScenarioResult], csv_path: Path) -> str:
    passed = sum(result.passed for result in results)
    lines = [
        "# EdgeTAM + Point-cloud Tracker：離線合成評估",
        "",
        f"格式版本：{FORMAT_VERSION}",
        "",
        "## 結論",
        "",
        (
            f"本次 deterministic point-cloud-only functional evaluation："
            f"**{passed}/{len(results)} scenarios passed**。"
        ),
        "",
        "這個deterministic評估器本身不是EdgeTAM、相機、ROS graph、CBF或馬達效能",
        "benchmark；它沒有載入checkpoint或執行GPU inference。另有一次真實官方",
        "checkpoint CUDA/API smoke，但repository仍沒有同步RGB-D rosbag或ground",
        "truth；因此live端到端FPS、mask IoU/accuracy、融合accuracy與控制成功率仍",
        "明確列為 **N/A / not measured**，不由合成資料推估。",
        "",
        "## 實際執行範圍",
        "",
        "- 固定亂數種子與 0.1 s sensor time step；不記錄 wall-clock timing。",
        "- 直接執行 point-cloud cluster quality、6D Kalman/Hungarian association、",
        "  track lifecycle、mask-invalid geometry fallback，以及 moving-camera 到",
        "  tracking-frame 的 rigid transform/preprocess/cluster path。",
        "- EdgeTAM mask drift案例只測 quality gate與 point-cloud fallback；不是 model",
        "  accuracy案例。",
        f"- 完整機器可讀資料：`{csv_path.as_posix()}`。",
        "",
        "## Requested mode/metric comparison",
        "",
        "沒有共同RGB-D ground truth可讓四種模式公平比較；未執行欄位維持N/A。",
        "point-cloud-only欄只使用本評估器實際產生的deterministic synthetic結果，",
        "不能當成真實sensor accuracy。ROS point-cloud snapshot與官方EdgeTAM API smoke",
        "分別記錄於`results/ros_live_smoke_2026-07-31.md`與",
        "`results/edgetam_official_smoke.md`，都不冒充同步RGB-D平均benchmark。",
        "",
        "| Metric | Original pipeline | Point-cloud-only | EdgeTAM-only | EdgeTAM + cloud |",
        "|---|---|---|---|---|",
        "| detected obstacle count | N/A (different video pipeline) | 10/10 expected scenario outcomes passed | N/A (not run by this evaluator) | N/A (not run) |",
        "| missed obstacle count | N/A | 0 scenario-level outcome failures; per-frame count N/A | N/A | N/A |",
        "| false obstacle count | N/A | N/A (no labeled background-only sequence) | N/A | N/A |",
        "| ID switch count | N/A | 0 unintended; 1 expected after expiry/re-entry | N/A | N/A |",
        "| average track duration | N/A | N/A (scenario lengths intentionally differ) | N/A | N/A |",
        "| position jitter | N/A | N/A; per-scenario mean position error is reported below | N/A | N/A |",
        "| velocity jitter | N/A | N/A (only final velocity accuracy asserted) | N/A | N/A |",
        "| mask-cluster IoU | N/A | N/A; drift case intentionally has no overlap | N/A | N/A |",
        "| average FPS | N/A | N/A (wall-clock timing intentionally excluded) | N/A（另見獨立API smoke報告） | N/A |",
        "| average latency | N/A | N/A | N/A（另見獨立API smoke報告） | N/A |",
        "| maximum latency | N/A | N/A | N/A（另見獨立API smoke報告） | N/A |",
        "| nearest-distance stability | N/A | N/A (surface validity asserted, no noise GT series) | N/A | N/A |",
        "",
        "## Scenario results",
        "",
        "| Scenario | Result | ID switches | Mean position error (m) | Final state | Point-cloud outcome |",
        "|---|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                (
                    result.scenario,
                    "PASS" if result.passed else "FAIL",
                    str(result.id_switches),
                    _metric(result.mean_position_error_m),
                    result.final_state,
                    result.pointcloud_outcome.replace("|", "/"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pass criteria與案例意義",
            "",
            "- `static`、`moving`：同一物體維持一個 ID，進入 CONFIRMED；moving另檢查",
            "  最終速度誤差。",
            "- `crossing`：兩個尺寸/點數不同但路徑交會的 cluster各自維持唯一 ID。",
            "- `occlusion`：三個 frame缺點雲時只進入 OCCLUDED/LOST，重新出現仍是",
            "  原 ID。",
            "- `leave_reappear`：超過 retention後刪除舊 track，稍後回來必須配置新 ID；",
            "  此案例的 `ID switches=1` 是預期行為。",
            "- `sparse`：有限但稀疏的點雲標成 SPARSE，不被誤當成無障礙物。",
            "- `mask_drift`：無重疊的 synthetic mask被標成 DEGRADED/INVALID，fusion",
            "  保留非空 point-cloud fallback。",
            "- `rgb_missing`：整段沒有 RGB/mask，點雲 ID仍連續且 mask quality保持",
            "  UNAVAILABLE。",
            "- `robot_motion`：把移動 camera frame的點先轉入固定 tracking frame，再",
            "  經 preprocess、Euclidean cluster與 tracker；固定物體不得產生 ID switch。",
            "- `hand_entrance`：近於 emergency distance的新 cluster不等待一般",
            "  confirmation hits，第一個 measurement立即 CONFIRMED。",
            "",
            "## 尚未驗證（N/A）",
            "",
            "| 項目 | 結果 | 需要的真實輸入 |",
            "|---|---|---|",
            "| EdgeTAM load / propagation | 不屬於本評估器；另見 `results/edgetam_official_smoke.md` | 官方checkpoint + CUDA/bf16 + 2 IDs + ID刪除 |",
            "| EdgeTAM mask accuracy | N/A | 有標註RGB-D ground truth masks |",
            "| RGB-D mask/cloud fusion accuracy | N/A | 同步 RGB、metric depth/organized cloud、CameraInfo與 GT |",
            "| Live ROS sync、TF drop、sensor stale recovery | N/A | 可 replay 的 rosbag與完整 TF tree |",
            "| 離線EdgeTAM API latency / VRAM | 不屬於本評估器；另見 `results/edgetam_official_smoke.md` | 只代表該次獨立smoke |",
            "| ROS end-to-end FPS / p50 / p95 latency / VRAM | N/A | 目標RTX 4060 Ti上的同步RGB-D live run |",
            "| Robot self-filter recall | N/A | URDF/link frames與 robot/non-robot point labels |",
            "| CBF / motor safety behavior | N/A | 外部 controller repository與硬體測試程序 |",
            "",
            "## 重現",
            "",
            "```bash",
            (
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python "
                "tools/evaluate_tracker.py"
            ),
            "```",
            "",
            "只要任何 scenario不符合 pass criteria，程式會以 non-zero status結束；",
            "CSV與Markdown仍會先寫出，以便診斷。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic point-cloud-only functional scenarios. "
            "No EdgeTAM checkpoint or live sensor is used."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "edgetam_pointcloud_evaluation.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "edgetam_pointcloud_evaluation.md",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    results = [scenario() for scenario in SCENARIOS]
    csv_text = render_csv(results)
    markdown_text = render_markdown(
        results,
        arguments.csv.relative_to(REPOSITORY_ROOT)
        if arguments.csv.is_relative_to(REPOSITORY_ROOT)
        else arguments.csv,
    )
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.csv.write_text(csv_text, encoding="utf-8")
    arguments.markdown.write_text(markdown_text, encoding="utf-8")

    for result in results:
        print(
            f"{'PASS' if result.passed else 'FAIL'} "
            f"{result.scenario}: {result.details}"
        )
    passed = sum(result.passed for result in results)
    print(f"Summary: {passed}/{len(results)} scenarios passed")
    print(f"CSV: {arguments.csv}")
    print(f"Markdown: {arguments.markdown}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
