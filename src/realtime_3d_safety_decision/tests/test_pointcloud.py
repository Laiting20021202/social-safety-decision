import numpy as np
import cv2
import pytest

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.apriltag_calibration import AprilTagScaleCalibrator
from realtime_safety.pipeline.pointcloud import (
    ReferenceDepthCalibrator,
    depth_to_pointmap,
    relative_inverse_depth,
    voxel_downsample,
)
from realtime_safety.types import PointCloudFrame


def _tag_cloud(frame_index: int = 0) -> tuple[np.ndarray, PointCloudFrame]:
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 120)
    image[60:180, 60:180] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    u, v = np.meshgrid(
        np.arange(240, dtype=np.float32), np.arange(240, dtype=np.float32)
    )
    # The reconstructed tag is 0.16 m wide although its physical width is
    # 0.08 m, so the expected global correction is approximately 0.5x.
    pointmap = np.stack(
        (u * (0.16 / 120.0), np.ones_like(u), -v * (0.16 / 120.0)),
        axis=-1,
    ).astype(np.float32)
    points = pointmap[::12, ::12].reshape(-1, 3).copy()
    cloud = PointCloudFrame(
        points=points,
        colors=np.full((len(points), 3), 128, np.uint8),
        confidence=np.ones(len(points), np.float32),
        pointmap=pointmap,
        frame_index=frame_index,
        timestamp=float(frame_index),
        anchor_frame_index=0,
        inference_ms=1.0,
        valid=True,
        source="synthetic",
    )
    return image, cloud


def test_apriltag_calibrator_locks_known_8cm_square_and_holds_scale() -> None:
    config = ReconstructionConfig(
        apriltag_enabled=True,
        apriltag_size_m=0.08,
        apriltag_detection_interval=1,
        apriltag_scale_ema_alpha=0.2,
        apriltag_hold_frames=3,
    )
    calibrator = AprilTagScaleCalibrator(config)
    image, cloud = _tag_cloud()
    calibrated = cloud
    for frame_index in range(config.apriltag_warmup_detections):
        image, cloud = _tag_cloud(frame_index)
        calibrated = calibrator.calibrate(image, cloud)

    assert calibrated.apriltag_locked
    assert calibrated.apriltag_id == 7
    assert calibrated.apriltag_scale_correction == pytest.approx(0.5, abs=0.02)
    edges = np.linalg.norm(
        calibrated.apriltag_corners_xyz
        - np.roll(calibrated.apriltag_corners_xyz, -1, axis=0),
        axis=1,
    )
    assert np.median(edges) == pytest.approx(0.08, abs=0.003)

    _, hidden = _tag_cloud(frame_index=config.apriltag_warmup_detections)
    blank = np.full_like(image, 255)
    held = calibrator.calibrate(blank, hidden)
    assert held.apriltag_locked
    assert held.apriltag_age_frames == 1
    assert held.metric_scale == pytest.approx(calibrated.metric_scale)


def test_depth_projection_coordinate_convention() -> None:
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    pointmap = depth_to_pointmap(depth, focal_px=2.0, principal_point=(1.0, 1.0))
    np.testing.assert_allclose(pointmap[1, 1], [0.0, 2.0, 0.0])
    assert pointmap[1, 2, 0] > 0.0
    assert pointmap[0, 1, 2] > 0.0


def test_relative_depth_is_finite_positive() -> None:
    inverse = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    depth = relative_inverse_depth(inverse)
    assert np.isfinite(depth).all()
    assert (depth > 0).all()
    assert np.isclose(np.median(depth), 3.0, rtol=0.05)


def test_voxel_and_max_count_are_bounded() -> None:
    points = np.stack((np.linspace(0, 1, 1000), np.zeros(1000), np.zeros(1000)), axis=-1)
    colors = np.full((1000, 3), 120, dtype=np.uint8)
    confidence = np.ones(1000, dtype=np.float32)
    result, result_colors, result_confidence = voxel_downsample(points, colors, confidence, 0.1, 5)
    assert result.shape == (5, 3)
    assert result_colors.shape == (5, 3)
    assert result_confidence.shape == (5,)


def test_reference_depth_calibrator_recovers_metric_scale_and_rejects_occlusion() -> None:
    calibrator = ReferenceDepthCalibrator(
        target_depth_m=0.4,
        roi=(0.4, 0.3, 0.6, 0.7),
        percentile=20.0,
        warmup_frames=3,
        ema_alpha=0.1,
    )
    depth = np.full((100, 100), 3.0, dtype=np.float32)
    depth[30:70, 40:60] = 1.0

    for _ in range(3):
        scale = calibrator.update(depth)

    assert calibrator.ready
    assert np.isclose(scale, 0.4)
    assert np.isclose(calibrator.observed_depth, 1.0)

    occluded = depth.copy()
    occluded[30:70, 40:60] = 0.2
    assert np.isclose(calibrator.update(occluded), 0.4)


def test_reference_depth_calibrator_can_freeze_scale_after_warmup() -> None:
    calibrator = ReferenceDepthCalibrator(
        target_depth_m=0.4,
        roi=(0.4, 0.3, 0.6, 0.7),
        percentile=20.0,
        warmup_frames=2,
        ema_alpha=0.0,
    )
    initial = np.ones((100, 100), dtype=np.float32)
    assert np.isclose(calibrator.update(initial), 0.4)
    assert np.isclose(calibrator.update(initial), 0.4)
    assert calibrator.ready

    drifted = np.full((100, 100), 1.1, dtype=np.float32)
    assert np.isclose(calibrator.update(drifted), 0.4)


def test_reference_depth_calibrator_slowly_reanchors_sustained_global_drift() -> None:
    calibrator = ReferenceDepthCalibrator(
        target_depth_m=0.4,
        roi=(0.25, 0.25, 0.75, 0.75),
        percentile=50.0,
        warmup_frames=3,
        ema_alpha=0.2,
        adaptation_frames=4,
        max_update_fraction=0.05,
    )
    startup = np.full((80, 80), 1.0, dtype=np.float32)
    for _ in range(3):
        assert np.isclose(calibrator.update(startup), 0.4)

    # A persistent 15% raw-depth shrink should ultimately restore 0.40 m,
    # but the first few corroborating frames must not jump the whole cloud.
    drifted = np.full((80, 80), 0.85, dtype=np.float32)
    first = calibrator.update(drifted)
    assert np.isclose(first, 0.4)
    for _ in range(39):
        final = calibrator.update(drifted)

    expected = 0.4 / 0.85
    assert first < final < expected
    assert np.isclose(final, expected, rtol=0.025)


def test_reference_depth_calibrator_ignores_temporary_full_roi_occlusion() -> None:
    calibrator = ReferenceDepthCalibrator(
        target_depth_m=0.4,
        roi=(0.25, 0.25, 0.75, 0.75),
        percentile=50.0,
        warmup_frames=2,
        ema_alpha=0.2,
        adaptation_frames=5,
    )
    reference = np.ones((80, 80), dtype=np.float32)
    calibrator.update(reference)
    calibrator.update(reference)
    initial_scale = calibrator.scale

    # A uniformly closer object is the depth-only worst case: spatial checks
    # cannot distinguish it from scale drift, so temporal persistence must.
    occluded = np.full((80, 80), 0.75, dtype=np.float32)
    for _ in range(3):
        assert np.isclose(calibrator.update(occluded), initial_scale)
    for _ in range(5):
        assert np.isclose(calibrator.update(reference), initial_scale)


def test_reference_depth_calibrator_rejects_local_persistent_foreground() -> None:
    calibrator = ReferenceDepthCalibrator(
        target_depth_m=0.4,
        roi=(0.25, 0.25, 0.75, 0.75),
        percentile=20.0,
        warmup_frames=2,
        ema_alpha=0.2,
        adaptation_frames=3,
    )
    reference = np.ones((80, 80), dtype=np.float32)
    calibrator.update(reference)
    calibrator.update(reference)
    initial_scale = calibrator.scale

    foreground = reference.copy()
    foreground[25:55, 25:55] = 0.75
    for _ in range(20):
        assert np.isclose(calibrator.update(foreground), initial_scale)
