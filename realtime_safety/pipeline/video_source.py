from __future__ import annotations

import threading
import time
from enum import Enum
from pathlib import Path

import cv2

from realtime_safety.types import FramePacket


class PlaybackState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class VideoSource:
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, source: str | int, loop: bool = False, playback_speed: float = 1.0) -> None:
        self.source = self._normalize_source(source)
        self.loop = loop
        self.playback_speed = max(float(playback_speed), 0.05)
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._state = PlaybackState.STOPPED
        self._frame_index = -1

    @staticmethod
    def _normalize_source(source: str | int) -> str | int:
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return int(source)
        value = str(source)
        if value.lower().startswith(("rtsp://", "rtsps://")):
            return value
        suffix = Path(value).suffix.lower()
        if suffix not in VideoSource.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {suffix or value}")
        if not Path(value).is_file():
            raise FileNotFoundError(value)
        return value

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def is_live(self) -> bool:
        return isinstance(self.source, int) or str(self.source).lower().startswith(("rtsp://", "rtsps://"))

    def open(self) -> None:
        with self._lock:
            self.close()
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open video source: {self.source}")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
            self._frame_index = -1
            self._state = PlaybackState.RUNNING

    def pause(self) -> None:
        with self._lock:
            if self._state == PlaybackState.RUNNING:
                self._state = PlaybackState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state == PlaybackState.PAUSED:
                self._state = PlaybackState.RUNNING

    def stop(self) -> None:
        with self._lock:
            self._state = PlaybackState.STOPPED

    def restart(self) -> None:
        with self._lock:
            if self._capture is None:
                self.open()
                return
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_index = -1
            self._state = PlaybackState.RUNNING

    def seek(self, seconds: float) -> None:
        if self.is_live:
            return
        with self._lock:
            if self._capture is None:
                raise RuntimeError("Video source is not open")
            self._capture.set(cv2.CAP_PROP_POS_MSEC, max(seconds, 0.0) * 1000.0)
            self._frame_index = max(int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, -1)

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = max(float(speed), 0.05)

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._state != PlaybackState.RUNNING or self._capture is None:
                return None
            ok, bgr = self._capture.read()
            if not ok:
                if self.loop and not self.is_live:
                    self.restart()
                    ok, bgr = self._capture.read()
                if not ok:
                    self._state = PlaybackState.ENDED
                    return None
            self._frame_index += 1
            capture_ts = time.perf_counter()
            fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            fps = fps if fps > 0.0 else 0.0
            reported_ts = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if self.is_live:
                source_ts = capture_ts
            elif reported_ts > 0.0 or self._frame_index == 0:
                source_ts = max(reported_ts, 0.0)
            else:
                source_ts = self._frame_index / fps if fps > 0 else float(self._frame_index)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = bgr.shape[:2]
            return FramePacket(
                frame_index=self._frame_index,
                source_timestamp=source_ts,
                capture_timestamp=capture_ts,
                bgr=bgr,
                rgb=rgb,
                original_fps=fps,
                original_width=width,
                original_height=height,
            )

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._state = PlaybackState.STOPPED

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
