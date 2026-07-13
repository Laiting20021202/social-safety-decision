from realtime_safety.config import load_config
from realtime_safety.scheduler import AdaptiveRealtimeController
from realtime_safety.types import PerformanceSnapshot


def test_repeated_latency_pressure_degrades_quality_not_safety_rate() -> None:
    config = load_config("realtime_fast")
    controller = AdaptiveRealtimeController(config)
    original_points = config.reconstruction.max_points
    safety_hz = config.safety.target_hz
    pressure = PerformanceSnapshot(
        display_fps=8.0,
        p95_latency_ms=450.0,
        queue_size=2,
        queue_capacity=2,
    )
    for _ in range(3):
        controller.observe(pressure)
    assert controller.profile == "DEGRADED"
    assert config.reconstruction.max_points < original_points
    assert config.safety.target_hz == safety_hz
