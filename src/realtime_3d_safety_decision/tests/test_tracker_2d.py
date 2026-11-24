import numpy as np

from realtime_safety.config import TrackingConfig
from realtime_safety.pipeline.tracker_2d import StableTracker2D
from realtime_safety.types import Detection2D


def detection(
    x: float,
    timestamp: float,
    class_name: str = "person",
) -> Detection2D:
    return Detection2D(
        bbox_xyxy=np.array([x, 10, x + 20, 50], dtype=np.float32),
        class_id=0,
        class_name=class_name,
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


def test_external_tracker_id_and_short_hold_remain_continuous() -> None:
    tracker = StableTracker2D(TrackingConfig(max_missing=4, visual_hold_updates=2))
    first_detection = detection(10, 1.0)
    first_detection.track_id = 42
    first = tracker.update_external([first_detection], 1.0)[0]

    tracker.update_external([], 1.1)
    held = tracker.predict_missing(1.1, max_missing=2)

    assert first.track_id == 42
    assert len(held) == 1
    assert held[0].track_id == 42
    assert held[0].track_missing == 1
    assert held[0].is_prediction

    recovered_detection = detection(12, 1.2)
    recovered_detection.track_id = 42
    recovered = tracker.update_external([recovered_detection], 1.2)[0]
    assert recovered.track_id == 42
    assert recovered.track_hits == 2
    assert recovered.track_missing == 0
    assert not recovered.is_prediction


def test_one_frame_class_error_does_not_split_or_relabel_track() -> None:
    tracker = StableTracker2D(TrackingConfig())
    first = tracker.update([detection(10, 1.0)], 1.0)[0]
    second = tracker.update([detection(11, 1.1)], 1.1)[0]
    noisy = tracker.update([detection(12, 1.2, "chair")], 1.2)[0]

    assert first.track_id == second.track_id == noisy.track_id
    assert noisy.class_name == "person"
