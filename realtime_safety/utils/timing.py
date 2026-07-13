from __future__ import annotations

import contextlib
import time
from collections import deque
from threading import Lock

import numpy as np


class RateMeter:
    def __init__(self, window: int = 120) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float | None = None) -> None:
        self._timestamps.append(float(timestamp if timestamp is not None else time.perf_counter()))

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class SampleWindow:
    def __init__(self, maxlen: int = 600) -> None:
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self._values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._values, 95)) if self._values else 0.0

    def values(self) -> list[float]:
        return list(self._values)


class CudaEventTimer(contextlib.AbstractContextManager):
    """CUDA-event duration with a wall-clock fallback."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.elapsed_ms = 0.0
        self._wall_start = 0.0
        self._start = None
        self._end = None

    def __enter__(self) -> "CudaEventTimer":
        self._wall_start = time.perf_counter()
        if self.enabled:
            try:
                import torch

                if torch.cuda.is_available():
                    self._start = torch.cuda.Event(enable_timing=True)
                    self._end = torch.cuda.Event(enable_timing=True)
                    self._start.record()
            except Exception:
                self._start = None
        return self

    def __exit__(self, *args: object) -> None:
        if self._start is not None and self._end is not None:
            import torch

            self._end.record()
            self._end.synchronize()
            self.elapsed_ms = float(self._start.elapsed_time(self._end))
        else:
            self.elapsed_ms = (time.perf_counter() - self._wall_start) * 1000.0


class LockedValue:
    def __init__(self, value):
        self._value = value
        self._lock = Lock()

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value
