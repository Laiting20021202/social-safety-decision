from __future__ import annotations

from pathlib import Path

from packages.frame_sources import HuggingFaceDatasetSource, ImageSequenceSource


def test_image_sequence_to_virtual_video(fixture_socialnav_root: Path) -> None:
    source = ImageSequenceSource(fixture_socialnav_root, interval_sec=0.25)

    scenario = source.get_scenario("demo_crossing")
    frame = source.get_frame("demo_crossing", 2)

    assert scenario.frame_count == 4
    assert scenario.duration_sec == 0.75
    assert frame.timestamp_sec == 0.5
    assert frame.original_timestamp is None
    assert frame.image_width == 640
    assert frame.image_height == 360


def test_socialnav_adapter_uses_local_mirror(fixture_socialnav_root: Path) -> None:
    source = HuggingFaceDatasetSource(
        local_repo=fixture_socialnav_root,
        revision="fixture",
        virtual_frame_interval_sec=0.5,
    )

    scenarios = source.list_scenarios()
    frame = source.get_frame("demo_crossing", 1)

    assert [scenario.scenario_id for scenario in scenarios] == ["demo_crossing"]
    assert frame.source_type == "huggingface"
    assert frame.timestamp_sec == 0.5
    assert "sample_with_bev_1.png" in str(source.get_frame_image_path("demo_crossing", 1))
