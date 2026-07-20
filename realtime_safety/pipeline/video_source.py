from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2

from realtime_safety.types import FramePacket


LOGGER = logging.getLogger(__name__)
AUTO_CAMERA_ALIASES = {"auto", "camera", "webcam", "usb", "usb-camera", "usb_camera"}


class CameraDetectionError(RuntimeError):
    """Raised when automatic camera discovery cannot find a readable stream."""


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    path: str
    name: str
    width: int
    height: int
    fps: float
    backend: str

    @property
    def description(self) -> str:
        resolution = f"{self.width}x{self.height}" if self.width > 0 and self.height > 0 else "unknown resolution"
        rate = f" @ {self.fps:.1f} FPS" if self.fps > 0 else ""
        return f"{self.name} ({self.path}, {resolution}{rate})"


def _candidate_camera_indices(max_index: int) -> list[int]:
    if sys.platform.startswith("linux"):
        indices = []
        for path in Path("/dev").glob("video[0-9]*"):
            suffix = path.name.removeprefix("video")
            capture_capability = _linux_capture_capability(path)
            if suffix.isdigit() and int(suffix) < max_index and capture_capability is not False:
                indices.append(int(suffix))
        return sorted(set(indices))
    return list(range(max(max_index, 0)))


def _linux_capture_capability(path: Path) -> bool | None:
    """Use udev metadata to skip known metadata-only V4L2 nodes."""

    try:
        device = path.stat().st_rdev
        data = Path(f"/run/udev/data/c{os.major(device)}:{os.minor(device)}").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in data.splitlines():
        if line.startswith("E:ID_V4L_CAPABILITIES="):
            return ":capture:" in line
    return None


def _open_camera_capture(index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def _camera_name(index: int) -> str:
    sysfs_name = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        name = sysfs_name.read_text(encoding="utf-8").strip()
    except OSError:
        name = ""
    return name or f"Webcam {index}"


def _camera_device(index: int, capture: cv2.VideoCapture, frame=None) -> CameraDevice:
    if frame is not None:
        height, width = frame.shape[:2]
    else:
        width = round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 0.0
    try:
        backend = str(capture.getBackendName())
    except (AttributeError, cv2.error):
        backend = "OpenCV"
    path = f"/dev/video{index}" if Path(f"/dev/video{index}").exists() else str(index)
    return CameraDevice(index, path, _camera_name(index), width, height, fps, backend)


def discover_cameras(max_index: int = 10, probe_frames: int = 3) -> list[CameraDevice]:
    """Return camera nodes that can produce a real color frame.

    Linux UVC devices commonly expose both a video node and a metadata node.
    Merely checking ``isOpened`` is therefore insufficient; discovery reads a
    small number of frames and only returns nodes that produce image data.
    """

    cameras: list[CameraDevice] = []
    for index in _candidate_camera_indices(max_index):
        capture = _open_camera_capture(index)
        try:
            if not capture.isOpened():
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = None
            for _ in range(max(int(probe_frames), 1)):
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size > 0:
                    frame = candidate
                    break
            if frame is None:
                continue
            cameras.append(_camera_device(index, capture, frame))
        finally:
            capture.release()
    LOGGER.info("Detected readable cameras: %s", [camera.description for camera in cameras])
    return cameras


def detect_camera(max_index: int = 10) -> CameraDevice:
    cameras = discover_cameras(max_index=max_index)
    if not cameras:
        raise CameraDetectionError(
            "No readable webcam was found. Check the USB connection and camera permissions, then try again."
        )
    return cameras[0]


class PlaybackState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class VideoSource:
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, source: str | int, loop: bool = False, playback_speed: float = 1.0) -> None:
        self.camera_info: CameraDevice | None = None
        if isinstance(source, str) and source.strip().lower() in AUTO_CAMERA_ALIASES:
            self.camera_info = detect_camera()
            source = self.camera_info.index
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
        value = str(source).strip()
        if value.startswith("/dev/video") and value.removeprefix("/dev/video").isdigit():
            return int(value.removeprefix("/dev/video"))
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
            capture = _open_camera_capture(self.source) if isinstance(self.source, int) else cv2.VideoCapture(self.source)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open video source: {self.source}")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.source, int):
                self.camera_info = _camera_device(self.source, capture)
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
            if self._capture is not None:
                self._capture.release()
                self._capture = None
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
