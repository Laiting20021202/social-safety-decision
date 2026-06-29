from fastapi import FastAPI

from packages.common_models import ModelInfo, ServiceHealth

app = FastAPI(title="social-safety-amr robopoint-service", version="0.1.0")


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(
        service="robopoint-service",
        status="degraded",
        ready=False,
        message="RoboPoint integration is planned for Phase 4.",
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        service="robopoint-service",
        model="wentaoyuan/RoboPoint",
        revision="unloaded",
        loaded=False,
        metadata={"blocking_reason": "Phase 4 integration not executed yet."},
    )
