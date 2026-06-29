from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from packages.common_models import Point2D, ZoneDefinition
from packages.frame_sources import HuggingFaceDatasetSource
from packages.overlay_renderer import ZoneStore
from services.dataset_service.app import create_app


def test_dataset_service_phase1_flow(fixture_socialnav_root: Path, tmp_path: Path) -> None:
    source = HuggingFaceDatasetSource(
        local_repo=fixture_socialnav_root,
        revision="fixture",
        virtual_frame_interval_sec=0.5,
    )
    app = create_app(frame_source=source, zone_store=ZoneStore(tmp_path))
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"

    datasets = client.get("/datasets").json()
    assert datasets[0]["dataset_id"] == "michaelmunje/SocialNav-SUB"

    scenarios = client.get("/datasets/michaelmunje/SocialNav-SUB/scenarios").json()
    assert scenarios[0]["scenario_id"] == "demo_crossing"

    state = client.post("/playback/start", json={"scenario_id": "demo_crossing"}).json()
    assert state["status"] == "playing"
    assert state["total_frames"] == 4

    state = client.post(
        "/playback/seek",
        json={"scenario_id": "demo_crossing", "frame_index": 2},
    ).json()
    assert state["frame_index"] == 2

    frame = client.get("/scenarios/demo_crossing/frames/2").json()
    assert frame["timestamp_sec"] == 1.0

    image_response = client.get("/scenarios/demo_crossing/frames/2/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    zone = ZoneDefinition(
        zone_id="manual-demo_crossing",
        scenario_id="demo_crossing",
        name="Danger zone",
        source="manual",
        polygon=[
            Point2D(x=200, y=200),
            Point2D(x=440, y=200),
            Point2D(x=440, y=340),
            Point2D(x=200, y=340),
        ],
        image_width=640,
        image_height=360,
        confidence=1.0,
    )
    saved = client.put("/zones/demo_crossing", json=zone.model_dump(mode="json")).json()
    assert saved["zone_id"] == "manual-demo_crossing"

    loaded = client.get("/zones/demo_crossing").json()
    assert loaded["name"] == "Danger zone"


def test_dataset_service_video_and_analysis_flow(
    fixture_socialnav_root: Path,
    tmp_path: Path,
) -> None:
    source = HuggingFaceDatasetSource(
        local_repo=fixture_socialnav_root,
        revision="fixture",
        virtual_frame_interval_sec=0.5,
    )
    app = create_app(frame_source=source, zone_store=ZoneStore(tmp_path))
    client = TestClient(app)

    video_info = client.get("/scenarios/demo_crossing/video-info").json()
    assert video_info["video_reference"] == "/scenarios/demo_crossing/video"
    assert video_info["source"] in {"generated_from_frames", "cached_mp4"}
    assert video_info["fps"] == 25.0

    video_response = client.get("/scenarios/demo_crossing/video")
    assert video_response.status_code == 200
    assert video_response.headers["content-type"].startswith("video/mp4")
    assert len(video_response.content) > 0

    packet = client.get(
        "/scenarios/demo_crossing/analysis",
        params={
            "timestamp_sec": 1.5,
            "prediction_horizon_sec": 3.0,
            "vqa_update_interval_sec": 2.0,
        },
    ).json()
    assert packet["scenario_id"] == "demo_crossing"
    assert packet["road"]["is_valid"]
    assert packet["road"]["source"] == "fixture_color_segmentation"
    assert packet["tracks"][0]["track_id"] == 1
    assert packet["motions"][0]["speed"] > 0
    assert packet["risk_zones"][0]["risk_polygon"]
    assert packet["metadata"]["formal_model_output"] is False

    road = ZoneDefinition(
        zone_id="road-demo_crossing",
        scenario_id="demo_crossing",
        name="Road / path calibration",
        source="manual_fallback",
        polygon=[
            Point2D(x=0, y=240),
            Point2D(x=640, y=240),
            Point2D(x=640, y=360),
            Point2D(x=0, y=360),
        ],
        image_width=640,
        image_height=360,
        confidence=1.0,
    )
    saved = client.put("/road/demo_crossing", json=road.model_dump(mode="json")).json()
    assert saved["source"] == "manual_fallback"

    packet = client.get("/scenarios/demo_crossing/analysis", params={"timestamp_sec": 0.0}).json()
    assert packet["road"]["source"] == "manual_fallback"
