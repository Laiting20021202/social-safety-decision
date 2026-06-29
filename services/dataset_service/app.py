from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from packages.common_models import ServiceHealth, ZoneDefinition
from packages.frame_sources import FrameSource, HuggingFaceDatasetSource
from packages.frame_sources.settings import DatasetSettings
from packages.overlay_renderer import ZoneStore
from services.dataset_service.analysis import AnalysisBuilder
from services.dataset_service.playback import (
    PlaybackConfigUpdate,
    PlaybackManager,
    SeekRequest,
    StartPlaybackRequest,
)
from services.dataset_service.video_cache import VideoCache


def build_default_source(settings: DatasetSettings | None = None) -> HuggingFaceDatasetSource:
    settings = settings or DatasetSettings()
    return HuggingFaceDatasetSource(
        dataset_id=settings.socialnav_dataset_id,
        revision=settings.socialnav_dataset_revision,
        cache_dir=settings.socialnav_cache_dir,
        local_repo=settings.socialnav_local_repo,
        virtual_frame_interval_sec=settings.socialnav_virtual_frame_interval_sec,
    )


def create_app(
    frame_source: FrameSource | None = None,
    zone_store: ZoneStore | None = None,
    settings: DatasetSettings | None = None,
) -> FastAPI:
    settings = settings or DatasetSettings()
    source = frame_source or build_default_source(settings)
    zones = zone_store or ZoneStore(settings.zone_config_dir)
    playback = PlaybackManager(source)
    video_cache = VideoCache()
    analysis = AnalysisBuilder(source, zones)

    app = FastAPI(title="social-safety-amr dataset-service", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.frame_source = source
    app.state.zone_store = zones
    app.state.playback = playback
    app.state.video_cache = video_cache
    app.state.analysis = analysis

    @app.get("/health", response_model=ServiceHealth)
    def health() -> ServiceHealth:
        return ServiceHealth(
            service="dataset-service",
            status="ok",
            ready=True,
            metadata={"app_mode": settings.app_mode},
        )

    @app.get("/ready", response_model=ServiceHealth)
    def ready() -> ServiceHealth:
        try:
            source.dataset_info()
            return ServiceHealth(service="dataset-service", status="ok", ready=True)
        except Exception as exc:  # pragma: no cover - defensive endpoint
            return ServiceHealth(
                service="dataset-service",
                status="error",
                ready=False,
                message=str(exc),
            )

    @app.get("/datasets")
    def datasets() -> list[dict[str, object]]:
        return [source.dataset_info().model_dump(mode="json")]

    @app.get("/datasets/{dataset_id:path}/splits")
    def dataset_splits(dataset_id: str) -> list[str]:
        _validate_dataset_id(source, dataset_id)
        return ["prompts"]

    @app.get("/datasets/{dataset_id:path}/scenarios")
    def dataset_scenarios(dataset_id: str) -> list[dict[str, object]]:
        _validate_dataset_id(source, dataset_id)
        return [scenario.model_dump(mode="json") for scenario in source.list_scenarios()]

    @app.get("/datasets/{dataset_id:path}")
    def dataset(dataset_id: str) -> dict[str, object]:
        info = source.dataset_info()
        if dataset_id not in {info.dataset_id, info.name}:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        return info.model_dump(mode="json")

    @app.get("/scenarios/{scenario_id}")
    def scenario(scenario_id: str) -> dict[str, object]:
        try:
            return source.get_scenario(scenario_id).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/scenarios/{scenario_id}/frames")
    def scenario_frames(scenario_id: str) -> list[dict[str, object]]:
        try:
            scenario_info = source.get_scenario(scenario_id)
            return [
                source.get_frame(scenario_id, index).model_dump(mode="json")
                for index in range(scenario_info.frame_count)
            ]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/scenarios/{scenario_id}/frames/{frame_index}")
    def scenario_frame(scenario_id: str, frame_index: int) -> dict[str, object]:
        try:
            return source.get_frame(scenario_id, frame_index).model_dump(mode="json")
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/scenarios/{scenario_id}/frames/{frame_index}/image")
    def scenario_frame_image(scenario_id: str, frame_index: int) -> FileResponse:
        try:
            path = source.get_frame_image_path(scenario_id, frame_index)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(Path(path))

    @app.get("/scenarios/{scenario_id}/video-info")
    def scenario_video_info(scenario_id: str) -> dict[str, object]:
        try:
            cached = video_cache.ensure_mp4(source, scenario_id)
            scenario_info = source.get_scenario(scenario_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "scenario_id": scenario_id,
            "video_reference": f"/scenarios/{scenario_id}/video",
            "source": cached.source,
            "fps": cached.fps,
            "duration_sec": scenario_info.duration_sec,
            "frame_count": scenario_info.frame_count,
            "generated": cached.generated,
        }

    @app.get("/scenarios/{scenario_id}/video")
    def scenario_video(scenario_id: str) -> FileResponse:
        try:
            cached = video_cache.ensure_mp4(source, scenario_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(Path(cached.path), media_type="video/mp4")

    @app.get("/scenarios/{scenario_id}/analysis")
    def scenario_analysis(
        scenario_id: str,
        timestamp_sec: float = Query(default=0.0, ge=0.0),
        prediction_horizon_sec: float = Query(default=3.0, gt=0.0, le=5.0),
        vqa_update_interval_sec: float = Query(default=2.0, gt=0.0, le=10.0),
    ) -> dict[str, object]:
        try:
            packet = analysis.packet(
                scenario_id=scenario_id,
                video_timestamp_sec=timestamp_sec,
                prediction_horizon_sec=prediction_horizon_sec,
                vqa_update_interval_sec=vqa_update_interval_sec,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return packet.model_dump(mode="json")

    @app.post("/playback/start")
    def playback_start(
        request: Annotated[StartPlaybackRequest | None, Body()] = None,
    ) -> dict[str, object]:
        try:
            return playback.start(request).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/playback/pause")
    def playback_pause() -> dict[str, object]:
        return playback.pause().model_dump(mode="json")

    @app.post("/playback/stop")
    def playback_stop() -> dict[str, object]:
        return playback.stop().model_dump(mode="json")

    @app.post("/playback/reset")
    def playback_reset() -> dict[str, object]:
        return playback.reset().model_dump(mode="json")

    @app.post("/playback/seek")
    def playback_seek(request: SeekRequest) -> dict[str, object]:
        try:
            return playback.seek(request).model_dump(mode="json")
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/playback/step")
    def playback_step(delta: Annotated[int, Body(embed=True)] = 1) -> dict[str, object]:
        try:
            return playback.step(delta=delta).model_dump(mode="json")
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/playback/config")
    def playback_config(update: PlaybackConfigUpdate) -> dict[str, object]:
        try:
            return playback.configure(update).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/playback/state")
    def playback_state() -> dict[str, object]:
        return playback.snapshot()

    @app.websocket("/playback/live")
    async def playback_live(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                snapshot = playback.snapshot()
                scenario_id = playback.state.scenario_id
                zone = (
                    zones.load(source.dataset_info().dataset_id, scenario_id)
                    if scenario_id
                    else None
                )
                snapshot["zone"] = zone.model_dump(mode="json") if zone else None
                await websocket.send_json(snapshot)
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return

    @app.get("/zones/{scenario_id}")
    def get_zone(scenario_id: str) -> dict[str, object] | None:
        zone = zones.load(source.dataset_info().dataset_id, scenario_id)
        return zone.model_dump(mode="json") if zone else None

    @app.put("/zones/{scenario_id}")
    def put_zone(scenario_id: str, zone: ZoneDefinition) -> dict[str, object]:
        if zone.scenario_id != scenario_id:
            raise HTTPException(status_code=400, detail="zone.scenario_id must match path")
        try:
            scenario_info = source.get_scenario(scenario_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if (
            zone.image_width
            and zone.image_height
            and (
                zone.image_width != scenario_info.image_width
                or zone.image_height != scenario_info.image_height
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="zone image dimensions do not match scenario",
            )
        zones.save(source.dataset_info().dataset_id, zone)
        return zone.model_dump(mode="json")

    @app.delete("/zones/{scenario_id}")
    def delete_zone(scenario_id: str) -> dict[str, bool]:
        return {"deleted": zones.delete(source.dataset_info().dataset_id, scenario_id)}

    @app.get("/road/{scenario_id}")
    def get_road(scenario_id: str) -> dict[str, object]:
        try:
            packet = analysis.packet(scenario_id=scenario_id, video_timestamp_sec=0.0)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return packet.road.model_dump(mode="json")

    @app.put("/road/{scenario_id}")
    def put_road(scenario_id: str, zone: ZoneDefinition) -> dict[str, object]:
        if zone.scenario_id != scenario_id:
            raise HTTPException(status_code=400, detail="zone.scenario_id must match path")
        if zone.polygon and len(zone.polygon) < 3:
            raise HTTPException(
                status_code=422,
                detail="road polygon must contain at least three points",
            )
        try:
            source.get_scenario(scenario_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        road_zone = zone.model_copy(
            update={
                "name": zone.name or "Road / path calibration",
                "source": "manual_fallback",
                "prompt": zone.prompt or "Mark the walkable path in front of the robot.",
                "metadata": {**zone.metadata, "semantic_role": "road_path_calibration"},
            }
        )
        zones.save(source.dataset_info().dataset_id, road_zone)
        return road_zone.model_dump(mode="json")

    @app.delete("/road/{scenario_id}")
    def delete_road(scenario_id: str) -> dict[str, bool]:
        return {"deleted": zones.delete(source.dataset_info().dataset_id, scenario_id)}

    return app


def _validate_dataset_id(source: FrameSource, dataset_id: str) -> None:
    info = source.dataset_info()
    if dataset_id not in {info.dataset_id, info.name}:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")


app = create_app()
