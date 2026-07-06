from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SAM3_REAL_TESTS") != "1",
    reason=(
        "Set RUN_SAM3_REAL_TESTS=1 with CUDA, HF_TOKEN, and SAM3_REAL_VIDEO_PATH "
        "to run real SAM 3 / SAM 3.1 checkpoint tests."
    ),
)


def test_docker_container_can_see_cuda() -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "compose.yaml",
            "-f",
            "compose.cuda.yaml",
            "--profile",
            "cuda-full",
            "run",
            "--rm",
            "sam3-service",
            "python3.12",
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(); "
                "print(torch.cuda.get_device_name(0)); "
                "print(torch.cuda.mem_get_info())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()


def test_facebook_sam3_image_checkpoint_loads() -> None:
    from services.sam3_service.runtime import Sam3RuntimeManager

    manager = Sam3RuntimeManager()
    state = manager.load_image_model()
    assert state.loaded
    assert state.repo_id == "facebook/sam3"
    manager.unload_image_model()


def test_facebook_sam31_multiplex_checkpoint_loads() -> None:
    from services.sam3_service.runtime import Sam3RuntimeManager

    manager = Sam3RuntimeManager()
    state = manager.load_video_model()
    assert state.loaded
    assert state.repo_id == "facebook/sam3.1"
    manager.unload_video_model()


def test_single_image_produces_true_mask() -> None:
    from services.sam3_service.runtime import Sam3RuntimeManager

    image_path = Path(os.environ["SAM3_REAL_IMAGE_PATH"])
    manager = Sam3RuntimeManager()
    result = manager.segment_image(image_path, ["person"])
    manager.unload_image_model()
    detection = result["detections"][0]
    assert detection["mask_rle"]
    assert not _polygon_is_bbox_rectangle(detection["mask_polygon"], detection["bounding_box"])


def test_video_session_person_prompt_tracks_object_id_across_10_frames() -> None:
    from services.sam3_service.runtime import Sam3RuntimeManager

    video_path = Path(os.environ["SAM3_REAL_VIDEO_PATH"]).resolve()
    manager = Sam3RuntimeManager()
    session = manager.start_video_session(str(video_path))
    session_id = str(session["session_id"])
    prompted = manager.video.add_prompt(session_id=session_id, prompt="person", frame_index=0)
    assert prompted["detections"]
    first_id = prompted["detections"][0]["track_id"]
    propagated = manager.video.propagate(
        session_id=session_id,
        start_frame_index=0,
        max_frame_num_to_track=10,
        include_frames=True,
    )
    frames = propagated["frames"]
    ids_by_frame = [
        {detection["track_id"] for detection in frame["detections"]}
        for frame in frames
        if frame["detections"]
    ]
    manager.video.close_session(session_id)
    manager.unload_video_model()

    assert len(ids_by_frame) >= 10
    assert all(first_id in frame_ids for frame_ids in ids_by_frame[:10])
    first_detection = frames[0]["detections"][0]
    assert not _polygon_is_bbox_rectangle(
        first_detection["mask_polygon"],
        first_detection["bounding_box"],
    )


def _polygon_is_bbox_rectangle(polygon: object, bbox: object) -> bool:
    if not isinstance(polygon, list) or not isinstance(bbox, tuple | list) or len(bbox) != 4:
        return False
    if len(polygon) != 4:
        return False
    points = {(round(point["x"], 3), round(point["y"], 3)) for point in polygon}
    x1, y1, x2, y2 = [round(float(value), 3) for value in bbox]
    return points == {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}
