import numpy as np

from realtime_safety.config import TrackingConfig
from realtime_safety.pipeline.tracker_2d import StableTracker2D
from realtime_safety.types import Detection2D


def detection(x: float, timestamp: float) -> Detection2D:
    return Detection2D(
        bbox_xyxy=np.array([x, 10, x + 20, 50], dtype=np.float32),
        class_id=0,
        class_name="person",
        confidence=0.9,
        centroid_xy=np.array([x + 10, 30], dtype=np.float32),
        timestamp=timestamp,
    )


def test_track_id_is_stable_and_velocity_uses_timestamp() -> None:
    tracker = StableTracker2D(TrackingConfig())
    first = tracker.update([detection(10, 1.0)], 1.0)[0]
    second = tracker.update([detection(20, 1.5)], 1.5)[0]
    assert first.track_id == second.track_id
    # EMA gain is 0.35, measured velocity is 20 px/s.
    assert np.isclose(second.velocity_xy[0], 7.0)


def test_short_occlusion_keeps_track() -> None:
    tracker = StableTracker2D(TrackingConfig(max_missing=3))
    first = tracker.update([detection(10, 1.0)], 1.0)[0]
    tracker.update([], 1.1)
    second = tracker.update([detection(11, 1.2)], 1.2)[0]
    assert first.track_id == second.track_id
