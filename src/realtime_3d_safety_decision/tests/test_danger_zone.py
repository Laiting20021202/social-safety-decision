import numpy as np

from realtime_safety.config import SafetyConfig
from realtime_safety.pipeline.danger_zone import DangerZonePredictor
from realtime_safety.types import BBox3D, Track3DState


def track(velocity: tuple[float, float, float], motion: str = "dynamic") -> Track3DState:
    position = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    return Track3DState(
        track_id=1,
        class_name="person",
        position_xyz=position,
        velocity_xyz=np.asarray(velocity, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6, dtype=np.float32) * 0.01,
        bbox3d=BBox3D(position - 0.2, position + 0.2),
        radius=0.3,
        hit_count=5,
        missing_count=0,
        last_timestamp=0.0,
        motion_state=motion,
        confidence=0.9,
    )


def test_danger_zone_follows_velocity_direction() -> None:
    zone = DangerZonePredictor(SafetyConfig()).predict([track((1.0, 0.0, 0.0))])[0]
    assert len(zone.predicted_positions) > 2
    assert np.all(np.diff(zone.predicted_positions[:, 0]) > 0)
    np.testing.assert_allclose(zone.predicted_positions[:, 1], 3.0)
    np.testing.assert_allclose(zone.predicted_direction, [1.0, 0.0, 0.0])


def test_static_obstacle_has_no_long_swept_volume() -> None:
    zone = DangerZonePredictor(SafetyConfig()).predict([track((0.0, 0.0, 0.0), motion="static")])[0]
    assert not zone.dynamic
    assert zone.predicted_positions.shape == (1, 3)
    assert zone.radii.shape == (1,)


def test_closing_obstacle_has_higher_risk() -> None:
    predictor = DangerZonePredictor(SafetyConfig())
    approaching = predictor.predict([track((0.0, -1.0, 0.0))])[0]
    leaving = predictor.predict([track((0.0, 1.0, 0.0))])[0]
    assert approaching.risk_score > leaving.risk_score
    assert approaching.closest_predicted_distance < leaving.closest_predicted_distance
