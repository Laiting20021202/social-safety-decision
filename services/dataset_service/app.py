from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from packages.common_models import DatasetInfo, ModelInfo, ServiceHealth, ZoneDefinition
from packages.frame_sources import FrameSource, HuggingFaceDatasetSource
from packages.frame_sources.settings import DatasetSettings
from packages.overlay_renderer import ZoneStore
from services.dataset_service.analysis import AnalysisBuilder
from services.dataset_service.bev_projection import profile_for_frame, render_bev_projection
from services.dataset_service.imported_video_source import ImportedVideoFrameSource
from services.dataset_service.playback import (
    PlaybackConfigUpdate,
    PlaybackManager,
    SeekRequest,
    StartPlaybackRequest,
)
from services.dataset_service.video_cache import VideoCache
from services.dataset_service.video_imports import ImportedVideoRecord, VideoImportStore


class Sam3VideoAnalysisRequest(BaseModel):
    prompt: str = "person"
    prompt_frame_index: int = Field(default=0, ge=0)
    max_frame_num_to_track: int | None = Field(default=None, ge=1)


class Sam3RoadAnalysisRequest(BaseModel):
    frame_index: int = Field(default=0, ge=0)
    prompts: list[str] = Field(
        default_factory=lambda: ["road", "sidewalk", "walkable path"],
        min_length=1,
    )


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
    video_cache = VideoCache(Path(settings.dataset_cache_dir) / "video_cache")
    video_imports = VideoImportStore(Path(settings.dataset_cache_dir) / "imported_videos")
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
    app.state.video_imports = video_imports
    app.state.imported_analysis_builders = {}
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

    @app.get("/readiness", response_model=ServiceHealth)
    def readiness() -> ServiceHealth:
        return ready()

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        dataset_info = source.dataset_info()
        return ModelInfo(
            service="dataset-service",
            model="precomputed-analysis-cache",
            revision=dataset_info.revision,
            loaded=True,
            metadata={
                "dataset_id": dataset_info.dataset_id,
                "sam3_service_url": analysis.sam3_client.base_url,
                "analysis_width": settings.sam3_analysis_width,
                "analysis_height": settings.sam3_analysis_height,
                "tracking_fps": settings.sam3_tracking_fps,
                "max_objects": settings.sam3_max_objects,
            },
        )

    @app.get("/runtime-info")
    def runtime_info() -> dict[str, object]:
        dataset_info = source.dataset_info()
        return {
            "service": "dataset-service",
            "dataset_id": dataset_info.dataset_id,
            "dataset_revision": dataset_info.revision,
            "sam3_service_url": analysis.sam3_client.base_url,
            "analysis_mode": "precomputed_timestamp_cache",
            "analysis_width": settings.sam3_analysis_width,
            "analysis_height": settings.sam3_analysis_height,
            "tracking_fps": settings.sam3_tracking_fps,
            "max_objects": settings.sam3_max_objects,
            "formal_model_output": False,
        }

    @app.get("/videos/imports")
    def imported_videos() -> list[dict[str, object]]:
        return [record.as_dict() for record in video_imports.list_records()]

    @app.get("/videos/imports/{video_id}")
    def imported_video(video_id: str) -> dict[str, object]:
        try:
            return video_imports.get_record(video_id).as_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/videos/imports/{video_id}/video")
    def imported_video_file(video_id: str) -> FileResponse:
        try:
            record = video_imports.get_record(video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(Path(record.path), media_type="video/mp4")

    @app.get("/videos/imports/{video_id}/frames/{frame_index}/image")
    def imported_video_frame_image(video_id: str, frame_index: int) -> FileResponse:
        try:
            builder = _imported_analysis_builder(
                video_id,
                video_imports=video_imports,
                zone_store=zones,
                settings=settings,
                builders=app.state.imported_analysis_builders,
            )
            path = builder.frame_source.get_frame_image_path(video_id, frame_index)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(Path(path))

    @app.get("/videos/imports/{video_id}/analysis")
    def imported_video_analysis(
        video_id: str,
        timestamp_sec: float = Query(default=0.0, ge=0.0),
        prediction_horizon_sec: float = Query(default=3.0, gt=0.0, le=5.0),
        vqa_update_interval_sec: float = Query(default=2.0, gt=0.0, le=10.0),
    ) -> dict[str, object]:
        try:
            builder = _imported_analysis_builder(
                video_id,
                video_imports=video_imports,
                zone_store=zones,
                settings=settings,
                builders=app.state.imported_analysis_builders,
            )
            packet = builder.packet(
                scenario_id=video_id,
                video_timestamp_sec=timestamp_sec,
                prediction_horizon_sec=prediction_horizon_sec,
                vqa_update_interval_sec=vqa_update_interval_sec,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return packet.model_dump(mode="json")

    @app.get("/videos/imports/{video_id}/bev-image")
    def imported_video_bev_image(
        video_id: str,
        timestamp_sec: float = Query(default=0.0, ge=0.0),
    ) -> FileResponse:
        try:
            builder = _imported_analysis_builder(
                video_id,
                video_imports=video_imports,
                zone_store=zones,
                settings=settings,
                builders=app.state.imported_analysis_builders,
            )
            path = _bev_projection_file(
                builder.frame_source,
                scenario_id=video_id,
                timestamp_sec=timestamp_sec,
                cache_root=Path(settings.dataset_cache_dir) / "bev_projection",
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(Path(path), media_type="image/png")

    @app.post("/videos/import")
    async def import_video(request: Request) -> dict[str, object]:
        content_type = request.headers.get("content-type", "")
        try:
            if "multipart/form-data" in content_type:
                form = await request.form()
                name = _optional_form_string(form.get("name"))
                dataset_name = _optional_form_string(form.get("dataset_name"))
                path_value = _optional_form_string(form.get("path") or form.get("mp4_path"))
                upload = form.get("file") or form.get("video")
                if upload is not None and hasattr(upload, "filename") and hasattr(upload, "read"):
                    filename = Path(str(upload.filename or "uploaded.mp4")).name
                    temp_path = video_imports.root / "uploads" / filename
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with temp_path.open("wb") as output:
                        while chunk := await upload.read(1024 * 1024):
                            output.write(chunk)
                    try:
                        return video_imports.import_file(
                            temp_path,
                            name=name or Path(filename).stem,
                            dataset_name=dataset_name,
                            source="upload",
                        ).as_dict()
                    finally:
                        temp_path.unlink(missing_ok=True)
                if not path_value:
                    raise HTTPException(
                        status_code=422,
                        detail="multipart import requires a file upload or path field",
                    )
                return video_imports.import_file(
                    path_value,
                    name=name,
                    dataset_name=dataset_name,
                    source="path",
                ).as_dict()

            payload = await request.json()
            if not isinstance(payload, dict):
                raise HTTPException(status_code=422, detail="JSON body must be an object")
            path_value = payload.get("path") or payload.get("mp4_path")
            if not isinstance(path_value, str) or not path_value.strip():
                raise HTTPException(status_code=422, detail="JSON import requires path")
            name_value = payload.get("name")
            dataset_value = payload.get("dataset_name")
            return video_imports.import_file(
                path_value,
                name=name_value if isinstance(name_value, str) else None,
                dataset_name=dataset_value if isinstance(dataset_value, str) else None,
                source="path",
            ).as_dict()
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/datasets")
    def datasets() -> list[dict[str, object]]:
        return [
            source.dataset_info().model_dump(mode="json"),
            *[item.model_dump(mode="json") for item in _imported_dataset_infos(video_imports)],
        ]

    @app.get("/datasets/{dataset_id:path}/splits")
    def dataset_splits(dataset_id: str) -> list[str]:
        if _is_imported_dataset(video_imports, dataset_id):
            return ["local_mp4"]
        _validate_dataset_id(source, dataset_id)
        return ["prompts"]

    @app.get("/datasets/{dataset_id:path}/scenarios")
    def dataset_scenarios(dataset_id: str) -> list[dict[str, object]]:
        imported_records = _imported_records_for_dataset(video_imports, dataset_id)
        if imported_records:
            return [
                ImportedVideoFrameSource(
                    record,
                    cache_root=video_imports.root / "analysis_frames" / record.video_id,
                    analysis_fps=settings.sam3_tracking_fps,
                )
                .get_scenario(record.video_id)
                .model_dump(mode="json")
                for record in imported_records
            ]
        _validate_dataset_id(source, dataset_id)
        return [scenario.model_dump(mode="json") for scenario in source.list_scenarios()]

    @app.get("/datasets/{dataset_id:path}")
    def dataset(dataset_id: str) -> dict[str, object]:
        imported = _imported_dataset_info(video_imports, dataset_id)
        if imported is not None:
            return imported.model_dump(mode="json")
        info = source.dataset_info()
        if dataset_id not in {info.dataset_id, info.name}:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        return info.model_dump(mode="json")

    @app.get("/scenarios/{scenario_id}")
    def scenario(scenario_id: str) -> dict[str, object]:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            return ImportedVideoFrameSource(
                imported_record,
                cache_root=video_imports.root / "analysis_frames" / imported_record.video_id,
                analysis_fps=settings.sam3_tracking_fps,
            ).get_scenario(imported_record.video_id).model_dump(mode="json")
        try:
            return source.get_scenario(scenario_id).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/scenarios/{scenario_id}/frames")
    def scenario_frames(scenario_id: str) -> list[dict[str, object]]:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            imported_source = ImportedVideoFrameSource(
                imported_record,
                cache_root=video_imports.root / "analysis_frames" / imported_record.video_id,
                analysis_fps=settings.sam3_tracking_fps,
            )
            scenario_info = imported_source.get_scenario(imported_record.video_id)
            return [
                imported_source.get_frame(imported_record.video_id, index).model_dump(mode="json")
                for index in range(scenario_info.frame_count)
            ]
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
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            return ImportedVideoFrameSource(
                imported_record,
                cache_root=video_imports.root / "analysis_frames" / imported_record.video_id,
                analysis_fps=settings.sam3_tracking_fps,
            ).get_frame(imported_record.video_id, frame_index).model_dump(mode="json")
        try:
            return source.get_frame(scenario_id, frame_index).model_dump(mode="json")
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/scenarios/{scenario_id}/frames/{frame_index}/image")
    def scenario_frame_image(scenario_id: str, frame_index: int) -> FileResponse:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            try:
                builder = _imported_analysis_builder(
                    imported_record.video_id,
                    video_imports=video_imports,
                    zone_store=zones,
                    settings=settings,
                    builders=app.state.imported_analysis_builders,
                )
                path = builder.frame_source.get_frame_image_path(
                    imported_record.video_id,
                    frame_index,
                )
            except (FileNotFoundError, IndexError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return FileResponse(Path(path))
        try:
            path = source.get_frame_image_path(scenario_id, frame_index)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(Path(path))

    @app.get("/scenarios/{scenario_id}/video-info")
    def scenario_video_info(scenario_id: str) -> dict[str, object]:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            return {
                "scenario_id": imported_record.video_id,
                "video_reference": f"/scenarios/{imported_record.video_id}/video",
                "source": "imported_mp4",
                "fps": imported_record.native_fps,
                "duration_sec": imported_record.duration_sec,
                "frame_count": imported_record.frame_count,
                "generated": False,
            }
        try:
            analysis.reset_if_scenario_changed(scenario_id)
            cached = video_cache.ensure_mp4(source, scenario_id, smooth_interpolate=True)
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
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            return FileResponse(Path(imported_record.path), media_type="video/mp4")
        try:
            cached = video_cache.ensure_mp4(source, scenario_id, smooth_interpolate=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(Path(cached.path), media_type="video/mp4")

    @app.get("/scenarios/{scenario_id}/bev-image")
    def scenario_bev_image(
        scenario_id: str,
        timestamp_sec: float = Query(default=0.0, ge=0.0),
    ) -> FileResponse:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            try:
                builder = _imported_analysis_builder(
                    imported_record.video_id,
                    video_imports=video_imports,
                    zone_store=zones,
                    settings=settings,
                    builders=app.state.imported_analysis_builders,
                )
                path = _bev_projection_file(
                    builder.frame_source,
                    scenario_id=imported_record.video_id,
                    timestamp_sec=timestamp_sec,
                    cache_root=Path(settings.dataset_cache_dir) / "bev_projection",
                )
            except (FileNotFoundError, IndexError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return FileResponse(Path(path), media_type="image/png")
        try:
            path = _bev_projection_file(
                source,
                scenario_id=scenario_id,
                timestamp_sec=timestamp_sec,
                cache_root=Path(settings.dataset_cache_dir) / "bev_projection",
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(Path(path), media_type="image/png")

    @app.get("/scenarios/{scenario_id}/analysis")
    def scenario_analysis(
        scenario_id: str,
        timestamp_sec: float = Query(default=0.0, ge=0.0),
        prediction_horizon_sec: float = Query(default=3.0, gt=0.0, le=5.0),
        vqa_update_interval_sec: float = Query(default=2.0, gt=0.0, le=10.0),
    ) -> dict[str, object]:
        imported_record = _imported_record_or_none(video_imports, scenario_id)
        if imported_record is not None:
            try:
                builder = _imported_analysis_builder(
                    imported_record.video_id,
                    video_imports=video_imports,
                    zone_store=zones,
                    settings=settings,
                    builders=app.state.imported_analysis_builders,
                )
                packet = builder.packet(
                    scenario_id=imported_record.video_id,
                    video_timestamp_sec=timestamp_sec,
                    prediction_horizon_sec=prediction_horizon_sec,
                    vqa_update_interval_sec=vqa_update_interval_sec,
                )
            except (FileNotFoundError, IndexError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return packet.model_dump(mode="json")
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

    @app.post("/scenarios/{scenario_id}/analysis/sam3-video")
    def scenario_sam3_video_analysis(
        scenario_id: str,
        request: Annotated[Sam3VideoAnalysisRequest | None, Body()] = None,
    ) -> dict[str, object]:
        request = request or Sam3VideoAnalysisRequest()
        try:
            source.get_scenario(scenario_id)
            cached = video_cache.ensure_mp4(
                source,
                scenario_id,
                target_fps=settings.sam3_tracking_fps,
                target_size=(settings.sam3_analysis_width, settings.sam3_analysis_height),
                crop_top_aspect_ratio=16 / 9,
            )
            return analysis.prepare_video_tracking(
                scenario_id=scenario_id,
                video_path=Path(cached.path).resolve(),
                prompt=request.prompt,
                prompt_frame_index=request.prompt_frame_index,
                video_fps=settings.sam3_tracking_fps,
                max_objects=settings.sam3_max_objects,
                max_frame_num_to_track=request.max_frame_num_to_track,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/scenarios/{scenario_id}/analysis/sam3-road")
    def scenario_sam3_road_analysis(
        scenario_id: str,
        request: Annotated[Sam3RoadAnalysisRequest | None, Body()] = None,
    ) -> dict[str, object]:
        request = request or Sam3RoadAnalysisRequest()
        try:
            source.get_frame(scenario_id, request.frame_index)
            return analysis.prepare_road_segmentation(
                scenario_id=scenario_id,
                frame_index=request.frame_index,
                prompts=request.prompts,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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


def _imported_dataset_infos(video_imports: VideoImportStore) -> list[DatasetInfo]:
    grouped: dict[str, list[str]] = {}
    for record in video_imports.list_records():
        grouped.setdefault(record.dataset_name, []).append(record.video_hash)
    return [
        DatasetInfo(
            dataset_id=name,
            name=name,
            revision=f"imported-{len(hashes)}",
            source_url="local-imported-mp4",
            cached=True,
            metadata={
                "source": "imported_videos",
                "video_count": len(hashes),
                "hashes": hashes,
            },
        )
        for name, hashes in sorted(grouped.items())
    ]


def _imported_dataset_info(
    video_imports: VideoImportStore,
    dataset_id: str,
) -> DatasetInfo | None:
    for info in _imported_dataset_infos(video_imports):
        if dataset_id in {info.dataset_id, info.name, _safe_path_part(info.name)}:
            return info
    return None


def _is_imported_dataset(video_imports: VideoImportStore, dataset_id: str) -> bool:
    return _imported_dataset_info(video_imports, dataset_id) is not None


def _imported_records_for_dataset(
    video_imports: VideoImportStore,
    dataset_id: str,
) -> list[ImportedVideoRecord]:
    records = [
        record
        for record in video_imports.list_records()
        if dataset_id in {
            record.dataset_name,
            _safe_path_part(record.dataset_name),
        }
    ]
    return sorted(records, key=lambda record: (-record.duration_sec, record.name.lower()))


def _imported_record_or_none(
    video_imports: VideoImportStore,
    video_id: str,
) -> ImportedVideoRecord | None:
    try:
        return video_imports.get_record(video_id)
    except FileNotFoundError:
        return None


def _imported_analysis_builder(
    video_id: str,
    *,
    video_imports: VideoImportStore,
    zone_store: ZoneStore,
    settings: DatasetSettings,
    builders: dict[str, AnalysisBuilder],
) -> AnalysisBuilder:
    existing = builders.get(video_id)
    if existing is not None:
        return existing
    record = video_imports.get_record(video_id)
    source = ImportedVideoFrameSource(
        record,
        cache_root=video_imports.root / "analysis_frames" / video_id,
        analysis_fps=settings.sam3_tracking_fps,
    )
    builder = AnalysisBuilder(source, zone_store)
    builder.lightweight_tracker.max_tracks = settings.sam3_max_objects
    builders.clear()
    builders[video_id] = builder
    return builder


def _bev_projection_file(
    frame_source: FrameSource,
    *,
    scenario_id: str,
    timestamp_sec: float,
    cache_root: Path,
) -> Path:
    scenario = frame_source.get_scenario(scenario_id)
    frame_index = _frame_index_for_timestamp(
        timestamp_sec,
        frame_count=scenario.frame_count,
        duration_sec=scenario.duration_sec,
    )
    frame = frame_source.get_frame(scenario_id, frame_index)
    profile = profile_for_frame(frame)
    if profile is None:
        raise FileNotFoundError(
            "BEV image projection requires a calibrated or estimated camera profile."
        )
    image_path = frame_source.get_frame_image_path(scenario_id, frame_index)
    dataset = _safe_path_part(frame.dataset_name or "dataset")
    revision = _safe_path_part(frame.dataset_revision or "revision")
    output_path = cache_root / dataset / revision / scenario_id / f"{frame_index:06d}.png"
    return render_bev_projection(
        Path(image_path),
        output_path,
        image_width=frame.image_width,
        image_height=frame.image_height,
        profile=profile,
    )


def _frame_index_for_timestamp(
    timestamp_sec: float,
    *,
    frame_count: int,
    duration_sec: float,
) -> int:
    if frame_count <= 1:
        return 0
    interval = duration_sec / (frame_count - 1) if duration_sec > 0 else 1.0 / 25.0
    return max(0, min(frame_count - 1, int(timestamp_sec / max(interval, 1e-6))))


def _safe_path_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def _optional_form_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


app = create_app()
