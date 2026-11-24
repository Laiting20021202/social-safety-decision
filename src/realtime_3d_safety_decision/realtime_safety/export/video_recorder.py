from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path: str | Path, fps: float = 20.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._lock = threading.Lock()

    def write(self, bgr: np.ndarray, source_fps: float = 0.0) -> None:
        with self._lock:
            if self._writer is None:
                height, width = bgr.shape[:2]
                fps = source_fps if source_fps > 0 else self.fps
                self._writer = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not self._writer.isOpened():
                    self._writer.release()
                    self._writer = None
                    raise RuntimeError(f"Cannot open annotated video writer: {self.path}")
            self._writer.write(np.ascontiguousarray(bgr))

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
