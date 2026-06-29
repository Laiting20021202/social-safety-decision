from fastapi import FastAPI

from packages.common_models import ServiceHealth

app = FastAPI(title="social-safety-amr geometry-service", version="0.1.0")


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(
        service="geometry-service",
        status="degraded",
        ready=False,
        message="Phase 2 service placeholder; core geometry helpers are available as packages.",
    )
