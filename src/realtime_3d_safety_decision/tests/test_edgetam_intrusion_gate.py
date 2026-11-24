from types import SimpleNamespace

from realtime_safety.edgetam_tracker.intrusion_gate import (
    SuddenIntrusionGate,
    SuddenIntrusionGateConfig,
)


def _track(
    track_id: int,
    *,
    speed: float,
    age: int,
    state: str = "CONFIRMED",
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        speed=speed,
        age_frames=age,
        state=state,
        velocity=(speed, 0.0, 0.0),
    )


def test_fixed_workspace_tracks_never_seed_an_intrusion() -> None:
    gate = SuddenIntrusionGate(
        SuddenIntrusionGateConfig(enabled=True, minimum_motion_hits=2)
    )

    assert gate.filter_tracks(
        [_track(1, speed=0.01, age=30)], baseline_ready=True
    ) == []
    assert gate.filter_tracks(
        [_track(1, speed=0.01, age=31)], baseline_ready=True
    ) == []


def test_new_moving_track_is_kept_after_it_stops() -> None:
    gate = SuddenIntrusionGate(
        SuddenIntrusionGateConfig(enabled=True, minimum_motion_hits=2)
    )

    assert gate.filter_tracks(
        [_track(7, speed=0.08, age=3)], baseline_ready=True
    ) == []
    accepted = gate.filter_tracks(
        [_track(7, speed=0.07, age=4)], baseline_ready=True
    )
    assert [track.track_id for track in accepted] == [7]
    held = gate.filter_tracks(
        [_track(7, speed=0.0, age=20, state="OCCLUDED")],
        baseline_ready=True,
    )
    assert [track.track_id for track in held] == [7]


def test_calibration_frames_cannot_seed_and_reset_previous_candidates() -> None:
    gate = SuddenIntrusionGate(
        SuddenIntrusionGateConfig(enabled=True, minimum_motion_hits=1)
    )
    assert gate.filter_tracks(
        [_track(2, speed=0.2, age=2)], baseline_ready=False
    ) == []
    assert not gate.accepted_track_ids


def test_gate_removes_deleted_ids() -> None:
    gate = SuddenIntrusionGate(
        SuddenIntrusionGateConfig(enabled=True, minimum_motion_hits=1)
    )
    assert gate.filter_tracks(
        [_track(3, speed=0.2, age=2)], baseline_ready=True
    )
    assert gate.filter_tracks(
        [_track(3, speed=0.0, age=3, state="DELETED")],
        baseline_ready=True,
    ) == []
    assert not gate.accepted_track_ids
