from __future__ import annotations

import threading
from dataclasses import replace

from realtime_safety.types import PipelineSnapshot


class GuiState:
    """Small synchronized latest-state store; it never retains frame history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = PipelineSnapshot()

    def publish(self, snapshot: PipelineSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def read(self) -> PipelineSnapshot:
        with self._lock:
            return replace(self._snapshot)
