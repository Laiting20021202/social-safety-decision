from fastapi import FastAPI

from packages.common_models import ModelInfo, ServiceHealth

app = FastAPI(title="social-safety-amr vqa-service", version="0.1.0")


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(
        service="vqa-service",
        status="degraded",
        ready=False,
        message="Temporal VQA providers are planned for Phase 5.",
    )


@app.get("/readiness", response_model=ServiceHealth)
def readiness() -> ServiceHealth:
    return health()


@app.get("/providers")
def providers() -> list[str]:
    return ["smolvlm", "qwen-planned"]


@app.get("/runtime-info")
def runtime_info() -> dict[str, object]:
    return {
        "service": "vqa-service",
        "loaded_runtime": None,
        "default_model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "formal_model_output": False,
        "blocking_reason": "Temporal VQA runtime is not loaded yet.",
    }



@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        service="vqa-service",
        model="unloaded",
        revision="unloaded",
        loaded=False,
        metadata={"blocking_reason": "Phase 5 integration not executed yet."},
    )
