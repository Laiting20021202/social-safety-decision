from __future__ import annotations

from social_bev.tracking import MultiObjectTracker
from social_bev.types import Detection


def person_box(x: float) -> Detection:
    return Detection(
        bbox=(x, 20.0, x + 30.0, 90.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        category="person",
    )


def tracker_config() -> dict[str, float | int]:
    return {
        "maximum_missed_frames": 3,
        "minimum_hits": 1,
        "history_length": 10,
        "iou_weight": 0.6,
        "distance_weight": 0.4,
        "max_center_distance_px": 80,
    }


def test_tracker_keeps_same_id_for_moving_box() -> None:
    tracker = MultiObjectTracker(tracker_config())
    ids = []
    for idx, x in enumerate([10, 15, 21, 28, 35]):
        tracks = tracker.update([person_box(float(x))], timestamp=idx * 0.1)
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_tracker_recovers_after_short_missing_detection() -> None:
    tracker = MultiObjectTracker(tracker_config())
    first = tracker.update([person_box(20.0)], timestamp=0.0)[0].track_id
    tracker.update([], timestamp=0.1)
    recovered = tracker.update([person_box(24.0)], timestamp=0.2)[0].track_id
    assert recovered == first


def test_empty_detections_do_not_crash() -> None:
    tracker = MultiObjectTracker(tracker_config())
    assert tracker.update([], timestamp=0.0) == []

