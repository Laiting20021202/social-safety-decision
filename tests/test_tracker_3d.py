import numpy as np

from realtime_safety.config import TrackingConfig
from realtime_safety.pipeline.tracker_3d import Tracker3D
from realtime_safety.types import BBox3D, ObstacleObservation3D


def observation(x: float, timestamp: float, track_id: int = 7) -> ObstacleObservation3D:
    position = np.array([x, 3.0, 0.0], dtype=np.float32)
    return ObstacleObservation3D(
        track_id=track_id,
        class_name="person",
        confidence=0.95,
        position_xyz=position,
        bbox3d=BBox3D(position - 0.2, position + 0.2),
        radius=0.3,
        point_count=100,
        timestamp=timestamp,
    )


def test_velocity_uses_real_timestamp_not_frame_number() -> None:
    config = TrackingConfig(dynamic_enter_speed=0.05, dynamic_exit_speed=0.02, minimum_dynamic_hits=1)
    fast = Tracker3D(config)
    slow = Tracker3D(config)
    fast.update([observation(0.0, 0.0)], 0.0)
    fast_state = fast.update([observation(1.0, 0.1)], 0.1)[0]
    slow.update([observation(0.0, 0.0)], 0.0)
    slow_state = slow.update([observation(1.0, 1.0)], 1.0)[0]
    assert fast_state.velocity_xyz[0] > slow_state.velocity_xyz[0] * 2.0


def test_predict_to_updates_position_without_measurement() -> None:
    tracker = Tracker3D(TrackingConfig())
    tracker.update([observation(0.0, 0.0)], 0.0)
    state = tracker.update([observation(1.0, 0.5)], 0.5)[0]
    future = tracker.predict_to(1.0)[0]
    assert future.position_xyz[0] > state.position_xyz[0]
