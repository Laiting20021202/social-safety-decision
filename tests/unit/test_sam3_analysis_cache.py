from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common_models import FramePacket
from packages.frame_sources import HuggingFaceDatasetSource
from packages.overlay_renderer import ZoneStore
from services.dataset_service.analysis import AnalysisBuilder, _track_observations_from_sam3


class _NoInferenceClient:
    configured = True

    def segment_image(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("segment_image must not run during GUI analysis requests")

    def close_session(self, session_id: str) -> dict[str, object]:
        return {"closed": session_id}


class _RecordingCloseClient(_NoInferenceClient):
    def __init__(self) -> None:
        self.closed: list[str] = []

    def close_session(self, session_id: str) -> dict[str, object]:
        self.closed.append(session_id)
        return {"closed": session_id}


class _RoadSegmentationClient(_NoInferenceClient):
    def segment_image(self, *_args: Any, **_kwargs: Any) -> dict[str, object]:
        return {
            "source": "sam3",
            "task": "image_concept_segmentation",
            "detections": [
                {
                    "class_name": "road",
                    "confidence": 0.91,
                    "label": "road",
                    "bounding_box": [8, 120, 190, 190],
                    "mask_rle": {"size": [200, 200], "counts": [20, 12, 80]},
                    "mask_polygon": [
                        {"x": 12, "y": 190},
                        {"x": 100, "y": 130},
                        {"x": 188, "y": 190},
                    ],
                }
            ],
        }


class _BlockedRoadSegmentationClient(_NoInferenceClient):
    def segment_image(self, *_args: Any, **_kwargs: Any) -> dict[str, object]:
        return {
            "source": "sam3_unavailable",
            "detections": [],
            "message": "SAM 3 image model unavailable: 401 gated repo; HF_TOKEN missing.",
        }


def test_analysis_packet_does_not_run_sam3_inference_on_request(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    builder = _builder(fixture_socialnav_root, tmp_path, _NoInferenceClient())

    packet = builder.packet("demo_crossing", video_timestamp_sec=1.0)

    assert packet.scenario_id == "demo_crossing"
    assert packet.metadata["sam3_precomputed"] is False


def test_precomputed_sam31_result_uses_official_object_id(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    builder = _builder(fixture_socialnav_root, tmp_path, _NoInferenceClient())
    builder._active_video_session_id = "sam31-session"
    builder._sam3_cache[("demo_crossing", 0)] = {
        "source": "sam3",
        "task": "sam3.1_video_tracking",
        "tracking_scope": "propagated",
        "detections": [
            {
                "track_id": 42,
                "object_id": 42,
                "class_name": "person",
                "confidence": 0.88,
                "bounding_box": [20, 30, 80, 140],
                "mask_rle": {"size": [360, 640], "counts": [100, 10, 230290]},
                "mask_polygon": [
                    {"x": 30, "y": 30},
                    {"x": 75, "y": 80},
                    {"x": 55, "y": 140},
                ],
                "centroid": {"x": 50, "y": 80},
                "ground_contact_point": {"x": 55, "y": 140},
                "segmentation_type": "binary_mask",
                "mask_format": "coco_rle_uncompressed",
                "label": "person",
            }
        ],
    }

    packet = builder.packet("demo_crossing", video_timestamp_sec=0.0)

    assert packet.tracks[0].track_id == 42
    assert packet.tracks[0].mask_rle is not None
    assert packet.tracks[0].metadata["official_object_id"] == 42
    assert packet.tracks[0].metadata["stable_track_id"] is True
    assert packet.system_status.tracking_status == "ok"
    assert packet.metadata["formal_model_output"] is True


def test_sam31_top_crop_coordinates_are_mapped_to_display_frame() -> None:
    frame = FramePacket(
        source_type="huggingface",
        dataset_name="SocialNav-SUB",
        dataset_revision="fixture",
        split="prompts",
        scenario_id="demo",
        frame_index=0,
        timestamp_sec=0.0,
        image_width=1280,
        image_height=2000,
        image_reference="/frames/0.png",
    )
    sam3_result = {
        "source": "sam3",
        "task": "sam3.1_video_tracking",
        "tracking_scope": "propagated",
        "detections": [
            {
                "track_id": 7,
                "object_id": 7,
                "class_name": "person",
                "confidence": 0.9,
                "bounding_box": [10, 20, 100, 160],
                "mask_rle": {"size": [360, 640], "counts": [10, 10]},
                "mask_polygon": [
                    {"x": 10, "y": 20},
                    {"x": 100, "y": 20},
                    {"x": 100, "y": 160},
                ],
                "centroid": {"x": 50, "y": 80},
                "ground_contact_point": {"x": 55, "y": 160},
                "segmentation_type": "binary_mask",
                "mask_format": "coco_rle_uncompressed",
                "label": "person",
            }
        ],
    }

    tracks = _track_observations_from_sam3(frame, sam3_result)

    assert len(tracks) == 1
    assert tracks[0].bounding_box == (20.0, 40.0, 200.0, 320.0)
    assert tracks[0].ground_contact_point is not None
    assert tracks[0].ground_contact_point.x == 110.0
    assert tracks[0].ground_contact_point.y == 320.0
    assert tracks[0].mask_polygon[1].x == 200.0
    assert tracks[0].metadata["coordinate_transform"]["source"] == "top_crop_width_fit"


def test_prompt_frame_segmentation_preview_is_not_cross_frame_tracking(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    builder = _builder(fixture_socialnav_root, tmp_path, _NoInferenceClient())
    builder._active_video_session_id = "sam31-session"
    builder._sam3_cache[("demo_crossing", 0)] = {
        "source": "sam3",
        "task": "sam3.1_video_tracking",
        "tracking_scope": "prompt_frame",
        "detections": [
            {
                "track_id": 5,
                "object_id": 5,
                "class_name": "person",
                "confidence": 0.88,
                "bounding_box": [20, 30, 80, 140],
                "mask_rle": {"size": [360, 640], "counts": [100, 10, 230290]},
                "mask_polygon": [
                    {"x": 30, "y": 30},
                    {"x": 75, "y": 80},
                    {"x": 55, "y": 140},
                ],
                "centroid": {"x": 50, "y": 80},
                "ground_contact_point": {"x": 55, "y": 140},
                "segmentation_type": "binary_mask",
                "mask_format": "coco_rle_uncompressed",
                "label": "person",
            }
        ],
    }

    packet = builder.packet("demo_crossing", video_timestamp_sec=0.0)

    assert len(packet.tracks) == 1
    assert packet.tracks[0].metadata["stable_track_id"] is False
    assert packet.system_status.tracking_status == "unavailable"
    assert packet.system_status.tracking_fps == 0.0
    assert packet.motions == []
    assert packet.risk_zones == []
    assert packet.metadata["formal_model_output"] is False
    assert packet.metadata["sam3_prompt_frame_preview"] is True
    assert packet.metadata["sam3_cross_frame_tracking"] is False
    assert "SAM3 Prompt-Frame Segmentation Preview" in packet.metadata["sam3_message"]
    assert (
        "Segmentation available, cross-frame tracking unavailable."
        in packet.system_status.message
    )


def test_prepare_road_segmentation_caches_true_mask(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    builder = _builder(fixture_socialnav_root, tmp_path, _RoadSegmentationClient())

    result = builder.prepare_road_segmentation("demo_crossing", frame_index=0)
    packet = builder.packet("demo_crossing", video_timestamp_sec=0.5)

    assert result["status"] == "ok"
    assert packet.road.is_valid is True
    assert packet.road.mask_rle is not None
    assert len(packet.road.polygon) == 3
    assert packet.road.source == "robopoint_sam3"
    assert packet.metadata["sam3_message"] == "SAM3 segmentation available."


def test_blocked_road_segmentation_message_is_reported(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    builder = _builder(fixture_socialnav_root, tmp_path, _BlockedRoadSegmentationClient())

    result = builder.prepare_road_segmentation("demo_crossing", frame_index=0)
    packet = builder.packet("demo_crossing", video_timestamp_sec=0.5)

    assert result["status"] == "degraded"
    assert "HF_TOKEN missing" in packet.metadata["sam3_message"]
    assert "401" in packet.system_status.message


def test_switching_scenario_closes_active_sam3_session(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    client = _RecordingCloseClient()
    builder = _builder(fixture_socialnav_root, tmp_path, client)
    builder._active_scenario_id = "old_scenario"
    builder._active_video_session_id = "old-session"
    builder._sam3_cache[("old_scenario", 0)] = {"source": "sam3"}
    builder._sam3_cache[("demo_crossing", 0)] = {"source": "sam3"}

    builder.reset_if_scenario_changed("demo_crossing")

    assert client.closed == ["old-session"]
    assert builder._active_scenario_id == "demo_crossing"
    assert builder._active_video_session_id is None
    assert ("old_scenario", 0) not in builder._sam3_cache
    assert ("demo_crossing", 0) in builder._sam3_cache


def _builder(
    fixture_socialnav_root: Path,
    tmp_path: Path,
    client: object,
) -> AnalysisBuilder:
    source = HuggingFaceDatasetSource(
        local_repo=fixture_socialnav_root,
        revision="fixture",
        virtual_frame_interval_sec=0.5,
    )
    return AnalysisBuilder(source, ZoneStore(tmp_path), sam3_client=client)  # type: ignore[arg-type]
