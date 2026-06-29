from __future__ import annotations

from pathlib import Path

from packages.frame_sources import ImageSequenceSource
from services.dataset_service.playback import PlaybackManager, SeekRequest, StartPlaybackRequest


def test_playback_seek_pause_step(fixture_socialnav_root: Path) -> None:
    manager = PlaybackManager(ImageSequenceSource(fixture_socialnav_root, interval_sec=0.5))

    state = manager.start(StartPlaybackRequest(scenario_id="demo_crossing"))
    assert state.status == "playing"
    assert state.total_frames == 4

    state = manager.pause()
    assert state.status == "paused"

    state = manager.seek(SeekRequest(scenario_id="demo_crossing", frame_index=2))
    assert state.frame_index == 2
    assert state.timestamp_sec == 1.0

    state = manager.step(delta=1)
    assert state.frame_index == 3

    state = manager.step(delta=1)
    assert state.frame_index == 3
    assert state.status == "ended"
