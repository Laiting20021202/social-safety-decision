from __future__ import annotations

import pytest

from packages.common_models import FramePacket, Point2D, ZoneDefinition


def test_frame_packet_schema_preserves_timestamp_metadata() -> None:
    packet = FramePacket(
        source_type="huggingface",
        dataset_name="michaelmunje/SocialNav-SUB",
        dataset_revision="f750caf",
        split="prompts",
        scenario_id="demo",
        frame_index=2,
        timestamp_sec=1.0,
        original_timestamp=None,
        fps=None,
        image_width=640,
        image_height=360,
        image_reference="/scenarios/demo/frames/2/image",
        metadata={"timestamp_source": "virtual_from_sequence"},
    )

    assert packet.timestamp_sec == 1.0
    assert packet.original_timestamp is None
    assert packet.metadata["timestamp_source"] == "virtual_from_sequence"


def test_zone_polygon_requires_three_points() -> None:
    with pytest.raises(ValueError):
        ZoneDefinition(
            zone_id="zone",
            scenario_id="demo",
            name="bad",
            source="manual",
            polygon=[Point2D(x=1, y=1), Point2D(x=2, y=2)],
        )


def test_manual_zone_schema() -> None:
    zone = ZoneDefinition(
        zone_id="manual-demo",
        scenario_id="demo",
        name="Danger zone",
        source="manual",
        polygon=[
            Point2D(x=10, y=10),
            Point2D(x=100, y=10),
            Point2D(x=80, y=100),
        ],
        image_width=640,
        image_height=360,
        confidence=1.0,
    )

    assert zone.source == "manual"
    assert zone.coordinate_type == "image"
    assert zone.confidence == 1.0
