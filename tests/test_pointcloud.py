import numpy as np

from realtime_safety.pipeline.pointcloud import depth_to_pointmap, relative_inverse_depth, voxel_downsample


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
