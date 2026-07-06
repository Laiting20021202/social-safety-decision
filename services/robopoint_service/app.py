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


@app.get("/readiness", response_model=ServiceHealth)
def readiness() -> ServiceHealth:
    return health()


@app.get("/runtime-info")
def runtime_info() -> dict[str, object]:
    return {
        "service": "robopoint-service",
        "loaded_runtime": None,
        "default_model": "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        "formal_model_output": False,
        "blocking_reason": "RoboPoint runtime is not loaded yet.",
    }


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        service="robopoint-service",
        model="wentaoyuan/RoboPoint",
        revision="unloaded",
        loaded=False,
        metadata={"blocking_reason": "Phase 4 integration not executed yet."},
    )
