import numpy as np

from realtime_safety.pipeline.ground_plane import GroundPlaneEstimator


def test_ransac_finds_horizontal_ground_with_outliers() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 800)
    y = rng.uniform(0.3, 10, 800)
    z = np.full(800, -1.2) + rng.normal(0, 0.015, 800)
    ground = np.stack((x, y, z), axis=-1)
    outliers = rng.uniform([-3, 0.3, -0.5], [3, 10, 2.0], (150, 3))
    estimate = GroundPlaneEstimator(distance_threshold=0.05, camera_height=1.2).estimate(np.r_[ground, outliers])
    assert estimate is not None
    assert estimate.confidence > 0.7
    assert estimate.coefficients[2] > 0.95
    assert np.isclose(estimate.height_at(np.array([0.0]), np.array([2.0]))[0], -1.2, atol=0.08)
