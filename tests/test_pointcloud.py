import numpy as np

from realtime_safety.pipeline.pointcloud import (
    ReferenceDepthCalibrator,
    depth_to_pointmap,
    relative_inverse_depth,
    voxel_downsample,
)


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
