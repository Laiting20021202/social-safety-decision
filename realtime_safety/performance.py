from __future__ import annotations

import threading

import psutil

from realtime_safety.types import PerformanceSnapshot
from realtime_safety.utils.gpu import gpu_info
from realtime_safety.utils.timing import RateMeter, SampleWindow


class PerformanceMonitor:
    STAGES = ("input", "display", "segmentation", "reconstruction", "safety")

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._lock = threading.Lock()
        self._rates = {stage: RateMeter() for stage in self.STAGES}
        self._latency = SampleWindow()

    def tick(self, stage: str, timestamp: float | None = None) -> None:
        with self._lock:
            self._rates[stage].tick(timestamp)

    def add_latency_ms(self, latency_ms: float) -> None:
        with self._lock:
            self._latency.add(latency_ms)

    def snapshot(self, dropped: int, queue_size: int, queue_capacity: int) -> PerformanceSnapshot:
        process = psutil.Process()
        gpu = gpu_info(self.device)
        with self._lock:
            return PerformanceSnapshot(
                input_fps=self._rates["input"].fps,
                display_fps=self._rates["display"].fps,
                segmentation_fps=self._rates["segmentation"].fps,
                reconstruction_fps=self._rates["reconstruction"].fps,
                safety_fps=self._rates["safety"].fps,
                average_latency_ms=self._latency.mean,
                p95_latency_ms=self._latency.p95,
                dropped_frames=dropped,
                queue_size=queue_size,
                queue_capacity=queue_capacity,
                ram_mb=process.memory_info().rss / 2**20,
                vram_used_mb=gpu.allocated_mb,
            )
