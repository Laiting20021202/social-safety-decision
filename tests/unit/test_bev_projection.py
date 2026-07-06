from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packages.common_models import FramePacket, Point2D
from services.dataset_service.bev_projection import (
    SCAND_AZURE_KINECT_ESTIMATED,
    profile_for_frame,
    project_pixel_to_bev_normalized,
    render_bev_projection,
)


def test_scand_profile_projects_lower_pixels_closer_to_robot() -> None:
    upper = project_pixel_to_bev_normalized(
        Point2D(x=640, y=430),
        image_width=1280,
        image_height=720,
        profile=SCAND_AZURE_KINECT_ESTIMATED,
    )
    lower = project_pixel_to_bev_normalized(
        Point2D(x=640, y=680),
        image_width=1280,
        image_height=720,
        profile=SCAND_AZURE_KINECT_ESTIMATED,
    )

    assert upper is not None
    assert lower is not None
    assert lower.y > upper.y
    assert 0.45 < lower.x < 0.55


def test_render_bev_projection_writes_png(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (1280, 720), (210, 214, 208))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 420, 1280, 720), fill=(150, 155, 150))
    draw.rectangle((580, 430, 700, 700), fill=(30, 40, 55))
    image.save(image_path)

    output = render_bev_projection(
        image_path,
        tmp_path / "bev.png",
        image_width=1280,
        image_height=720,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    with Image.open(output) as rendered:
        assert rendered.size == (1000, 420)


def test_profile_for_imported_video_uses_scand_estimate() -> None:
    frame = FramePacket(
        source_type="local_video",
        dataset_name="SCAND",
        dataset_revision="local",
        split="local_mp4",
        scenario_id="sample",
        frame_index=0,
        timestamp_sec=0.0,
        fps=5.0,
        image_width=1280,
        image_height=720,
        image_reference="/sample.jpg",
    )

    assert profile_for_frame(frame) == SCAND_AZURE_KINECT_ESTIMATED
