from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from packages.common_models import ModelInfo, ServiceHealth
from services.sam3_service.runtime import Sam3Runtime, write_upload_to_temp

app = FastAPI(title="social-safety-amr sam3-service", version="0.1.0")
runtime = Sam3Runtime()


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    status = runtime.status()
    return ServiceHealth(
        service="sam3-service",
        status="ok" if status.available else "degraded",
        ready=status.available,
        message=status.message,
        metadata={"backend": status.backend},
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    status = runtime.status()
    return ModelInfo(
        service="sam3-service",
        model="facebook/sam3",
        revision="runtime",
        loaded=status.available,
        metadata={"backend": status.backend, "message": status.message},
    )


@app.post("/segment-image")
async def segment_image(
    image: Annotated[UploadFile, File()],
    prompts: str = "person,bicycle,motorcycle,car,bus,truck,road,sidewalk,walkable path",
) -> dict[str, object]:
    data = await image.read()
    path = write_upload_to_temp(data, suffix=Path(image.filename or "frame.png").suffix or ".png")
    prompt_list = [prompt.strip() for prompt in prompts.split(",") if prompt.strip()]
    try:
        return runtime.segment_image(path, prompt_list)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)
