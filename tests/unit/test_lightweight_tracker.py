from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packages.frame_sources import ImageSequenceSource
from packages.overlay_renderer import ZoneStore
from services.dataset_service.analysis import AnalysisBuilder
from services.dataset_service.lightweight_tracker import LightweightVisionTracker


class _NoSam3Client:
    configured = False


def test_lightweight_tracker_keeps_id_across_frames(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    tracker = LightweightVisionTracker(max_tracks=6)

    first = tracker.observations_until("fast_scene", 0, source)
    later = tracker.observations_until("fast_scene", 4, source)

    first_ids = {track.track_id for track in first}
    later_ids = {track.track_id for track in later}
    assert first_ids & later_ids
    assert later[0].metadata["initializer"] == "lightweight_visual_tracker"
    assert later[0].mask_polygon == []
    assert later[0].bounding_box is not None


def test_analysis_uses_lightweight_tracker_when_sam_is_unavailable(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    builder = AnalysisBuilder(source, ZoneStore(tmp_path / "zones"), _NoSam3Client())

    packet = builder.packet("fast_scene", video_timestamp_sec=0.8)

    assert packet.tracks
    assert packet.metadata["tracking_mode"] == "lightweight_visual_tracker"
    assert packet.metadata["segmentation_status"] == "not_used_lightweight_boxes"
    assert packet.metadata["formal_model_output"] is False
    assert packet.system_status.tracking_status == "ok"
    assert packet.system_status.tracking_fps > 0
    assert packet.motions
    assert all(track.mask_polygon == [] for track in packet.tracks)


def _synthetic_source(tmp_path: Path) -> ImageSequenceSource:
    scenario_dir = tmp_path / "prompts" / "fast_scene"
    scenario_dir.mkdir(parents=True)
    for index in range(5):
        image = Image.new("RGB", (640, 360), (214, 218, 210))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 250, 640, 360), fill=(186, 188, 180))
        x = 170 + index * 18
        draw.rectangle((x, 106, x + 36, 242), fill=(28, 34, 42))
        draw.ellipse((x + 6, 78, x + 30, 104), fill=(40, 34, 30))
        draw.rectangle((420 - index * 8, 180, 480 - index * 8, 255), fill=(170, 44, 36))
        image.save(scenario_dir / f"frame_{index:03d}.png")
    return ImageSequenceSource(tmp_path, interval_sec=0.2)
