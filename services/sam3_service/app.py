from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from packages.common_models import ModelInfo, ServiceHealth
from services.sam3_service.runtime import Sam3RuntimeError, Sam3RuntimeManager, write_upload_to_temp

app = FastAPI(title="social-safety-amr sam3-service", version="0.2.0")
runtime = Sam3RuntimeManager()


class VideoSessionRequest(BaseModel):
    video_path: str | None = None
    resource_path: str | None = None
    scenario_id: str | None = None
    analysis_fps: float = Field(default=5.0, gt=0.0, le=5.0)
    max_objects: int = Field(default=6, ge=1, le=6)
    session_id: str | None = None

    def resolved_path(self) -> str:
        path = self.video_path or self.resource_path
        if not path:
            raise ValueError("video_path is required.")
        return path


class VideoPromptRequest(BaseModel):
    prompt: str = "person"
    frame_index: int = Field(default=0, ge=0)
    output_prob_thresh: float = Field(default=0.5, ge=0.0, le=1.0)


class VideoPropagateRequest(BaseModel):
    propagation_direction: Literal["both", "forward", "backward"] = "forward"
    start_frame_index: int | None = Field(default=None, ge=0)
    max_frame_num_to_track: int | None = Field(default=None, ge=1)
    output_prob_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    include_frames: bool = False


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    status = runtime.status()
    image_model = cast(dict[str, object], status["image_model"])
    video_model = cast(dict[str, object], status["video_model"])
    image_loaded = bool(image_model["loaded"])
    video_loaded = bool(video_model["loaded"])
    message = (
        "SAM 3 runtime idle; models are lazy-loaded."
        if not (image_loaded or video_loaded)
        else "SAM 3 model loaded."
    )
    return ServiceHealth(
        service="sam3-service",
        status="ok",
        ready=True,
        message=message,
        metadata=status,
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    status = runtime.status()
    image_model = cast(dict[str, object], status["image_model"])
    video_model = cast(dict[str, object], status["video_model"])
    return ModelInfo(
        service="sam3-service",
        model="facebook/sam3 + facebook/sam3.1",
        revision=str(status["sam3_package_commit"]),
        provider="facebookresearch/sam3",
        loaded=bool(image_model["loaded"]) or bool(video_model["loaded"]),
        metadata=status,
    )


@app.get("/readiness", response_model=ServiceHealth)
def readiness() -> ServiceHealth:
    return health()


@app.get("/runtime-info")
def runtime_info() -> dict[str, object]:
    return runtime.status()


@app.post("/models/image/load")
def load_image_model() -> dict[str, object]:
    try:
        return runtime.load_image_model().as_dict()
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.delete("/models/image")
def unload_image_model() -> dict[str, object]:
    return runtime.unload_image_model().as_dict()


@app.post("/models/video/load")
def load_video_model() -> dict[str, object]:
    try:
        return runtime.load_video_model().as_dict()
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.delete("/models/video")
def unload_video_model() -> dict[str, object]:
    return runtime.unload_video_model().as_dict()


@app.post("/cache/empty")
def empty_cache() -> dict[str, object]:
    from services.sam3_service.runtime import empty_cuda_cache, vram_usage

    empty_cuda_cache()
    return {"status": "ok", "vram": vram_usage()}


@app.post("/segment-image")
async def segment_image(
    image: Annotated[UploadFile, File()],
    prompts: str = "road,sidewalk,walkable path",
) -> dict[str, object]:
    data = await image.read()
    path = write_upload_to_temp(data, suffix=Path(image.filename or "frame.png").suffix or ".png")
    prompt_list = [prompt.strip() for prompt in prompts.split(",") if prompt.strip()]
    try:
        return runtime.segment_image(path, prompt_list)
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/video/sessions")
def start_video_session(request: VideoSessionRequest) -> dict[str, object]:
    try:
        return runtime.start_video_session(
            resource_path=request.resolved_path(),
            session_id=request.session_id,
            scenario_id=request.scenario_id,
            analysis_fps=request.analysis_fps,
            max_objects=request.max_objects,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.post("/video/sessions/{session_id}/prompts")
def add_video_prompt(session_id: str, request: VideoPromptRequest) -> dict[str, object]:
    try:
        return runtime.video.add_prompt(
            session_id=session_id,
            prompt=request.prompt,
            frame_index=request.frame_index,
            output_prob_thresh=request.output_prob_thresh,
        )
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.post("/video/sessions/{session_id}/propagate")
def propagate_video(
    session_id: str,
    request: Annotated[VideoPropagateRequest | None, Body()] = None,
) -> dict[str, object]:
    request = request or VideoPropagateRequest()
    try:
        return runtime.video.propagate(
            session_id=session_id,
            propagation_direction=request.propagation_direction,
            start_frame_index=request.start_frame_index,
            max_frame_num_to_track=request.max_frame_num_to_track,
            output_prob_thresh=request.output_prob_thresh,
            include_frames=request.include_frames,
        )
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.get("/video/sessions/{session_id}/frames/{frame_index}")
def video_frame_result(session_id: str, frame_index: int) -> dict[str, object]:
    try:
        return runtime.video.get_frame_result(session_id=session_id, frame_index=frame_index)
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.get("/video/sessions/{session_id}/status")
def video_session_status(session_id: str) -> dict[str, object]:
    try:
        return runtime.video.session_status(session_id)
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


@app.delete("/video/sessions/{session_id}")
def close_video_session(session_id: str) -> dict[str, object]:
    try:
        return runtime.video.close_session(session_id)
    except Sam3RuntimeError as exc:
        raise _http_error(exc) from exc


def _http_error(exc: Sam3RuntimeError) -> HTTPException:
    return HTTPException(
        status_code=503 if exc.degraded else 400,
        detail={
            "status": "degraded" if exc.degraded else "error",
            "message": str(exc),
        },
    )
