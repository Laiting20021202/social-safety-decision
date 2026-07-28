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
NETWORK_STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtsps://")
ROS2_STREAM_PREFIX = "ros2://"
NETWORK_RECONNECT_INITIAL_DELAY_SECONDS = 1.0
# Keep retries sparse enough not to accumulate abandoned web_video_server
# handlers, but short enough that a recovered camera never looks dead for a
# full minute. Open/read calls remain bounded separately below.
NETWORK_RECONNECT_MAX_DELAY_SECONDS = 10.0


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


def _open_network_capture(source: str) -> cv2.VideoCapture:
    """Open a live HTTP/RTSP stream with bounded FFmpeg timeouts when supported.

    Do not retry a failed FFmpeg open with OpenCV's default constructor.  That
    fallback drops the timeout parameters and can block the capture worker for
    roughly 30 seconds when a Wi-Fi camera disappears.
    """

    if hasattr(cv2, "CAP_FFMPEG"):
        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            # The Koch Wi-Fi path has occasionally paused for several seconds.
            # A short timeout churns web_video_server connections and can
            # eventually exhaust its stream handlers during an overnight run.
            params.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000))
        try:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            # Older OpenCV builds do not accept constructor parameters.  Keep
            # compatibility for those builds; current deployments take the
            # bounded path above.
            pass
    return cv2.VideoCapture(source)


def _open_ros2_capture(topic: str):
    from realtime_safety.ros2_bridge.image_subscriber import Ros2ImageCapture

    return Ros2ImageCapture(topic)


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
        self._ros_capture = None
        self._lock = threading.RLock()
        self._state = PlaybackState.STOPPED
        self._frame_index = -1
        self._last_reconnect_attempt = 0.0
        self._reconnect_delay = NETWORK_RECONNECT_INITIAL_DELAY_SECONDS

    @staticmethod
    def _normalize_source(source: str | int) -> str | int:
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return int(source)
        value = str(source).strip()
        if value.startswith("/dev/video") and value.removeprefix("/dev/video").isdigit():
            return int(value.removeprefix("/dev/video"))
        if value.lower().startswith(NETWORK_STREAM_PREFIXES):
            return value
        if value.lower().startswith(ROS2_STREAM_PREFIX):
            topic = value[len(ROS2_STREAM_PREFIX) :]
            if not topic.startswith("/") or any(char.isspace() for char in topic):
                raise ValueError(
                    f"ROS 2 image topic must be an absolute name without whitespace: {topic or value}"
                )
            return ROS2_STREAM_PREFIX + topic
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
        return isinstance(self.source, int) or self.is_remote_stream

    @property
    def is_network_stream(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith(NETWORK_STREAM_PREFIXES)

    @property
    def is_ros2_stream(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith(ROS2_STREAM_PREFIX)

    @property
    def is_remote_stream(self) -> bool:
        return self.is_network_stream or self.is_ros2_stream

    @property
    def ros2_topic(self) -> str:
        if not self.is_ros2_stream:
            raise ValueError(f"Not a ROS 2 image source: {self.source}")
        return str(self.source)[len(ROS2_STREAM_PREFIX) :]

    @property
    def is_connected(self) -> bool:
        with self._lock:
            if self.is_ros2_stream:
                return self._ros_capture is not None and self._ros_capture.is_connected
            return self._capture is not None and self._capture.isOpened()

    def _new_capture(self) -> cv2.VideoCapture:
        if isinstance(self.source, int):
            return _open_camera_capture(self.source)
        if self.is_network_stream:
            return _open_network_capture(self.source)
        return cv2.VideoCapture(self.source)

    def open(self) -> None:
        with self._lock:
            self.close()
            if self.is_ros2_stream:
                self._ros_capture = _open_ros2_capture(self.ros2_topic)
                self._ros_capture.start()
                self._frame_index = -1
                self._state = PlaybackState.RUNNING
                return
            capture = self._new_capture()
            if not capture.isOpened():
                capture.release()
                if self.is_network_stream:
                    # A live network source may be temporarily unavailable at
                    # startup.  Keep the pipeline alive so the capture worker
                    # can reconnect without requiring a GUI restart.
                    self._capture = None
                    self._frame_index = -1
                    self._last_reconnect_attempt = time.perf_counter()
                    self._reconnect_delay = NETWORK_RECONNECT_INITIAL_DELAY_SECONDS
                    self._state = PlaybackState.RUNNING
                    LOGGER.warning("Network video unavailable; waiting to reconnect: %s", self.source)
                    return
                raise RuntimeError(f"Cannot open video source: {self.source}")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.source, int):
                self.camera_info = _camera_device(self.source, capture)
            self._capture = capture
            self._frame_index = -1
            self._last_reconnect_attempt = 0.0
            self._reconnect_delay = NETWORK_RECONNECT_INITIAL_DELAY_SECONDS
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
            if self._ros_capture is not None:
                self._ros_capture.close()
                self._ros_capture = None
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._state = PlaybackState.STOPPED

    def restart(self) -> None:
        with self._lock:
            if self.is_remote_stream:
                self.open()
                return
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
            if self._state != PlaybackState.RUNNING:
                return None
            if self.is_ros2_stream:
                if self._ros_capture is None:
                    return None
                sample = self._ros_capture.read_latest()
                if sample is None:
                    return None
                bgr, ros_timestamp, fps = sample
                self._frame_index += 1
                capture_ts = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = bgr.shape[:2]
                return FramePacket(
                    frame_index=self._frame_index,
                    source_timestamp=ros_timestamp if ros_timestamp > 0.0 else capture_ts,
                    capture_timestamp=capture_ts,
                    bgr=bgr,
                    rgb=rgb,
                    original_fps=fps,
                    original_width=width,
                    original_height=height,
                )
            if self._capture is None:
                if not self._reconnect_network():
                    return None
            ok, bgr = self._capture.read()
            if not ok:
                if self.loop and not self.is_live:
                    self.restart()
                    ok, bgr = self._capture.read()
                elif self.is_network_stream:
                    # Return control immediately after a timed-out read.  The
                    # next capture iteration performs a bounded reconnect;
                    # doing both operations here doubles the visible stall.
                    self._capture.release()
                    self._capture = None
                    self._schedule_reconnect_backoff()
                    LOGGER.warning(
                        "Lost network video stream; retrying in %.0f seconds: %s",
                        self._reconnect_delay,
                        self.source,
                    )
                    return None
                if not ok:
                    if self.is_network_stream:
                        return None
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
            if self.is_network_stream:
                # A decoded frame proves that the new stream is genuinely
                # healthy. Merely opening a TCP connection is insufficient:
                # a hung web_video_server can accept without sending bytes.
                self._reconnect_delay = NETWORK_RECONNECT_INITIAL_DELAY_SECONDS
                self._last_reconnect_attempt = 0.0
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

    def _reconnect_network(self) -> bool:
        if not self.is_network_stream:
            return False
        now = time.perf_counter()
        if now - self._last_reconnect_attempt < self._reconnect_delay:
            return False
        self._last_reconnect_attempt = now
        if self._capture is not None:
            self._capture.release()
        capture = self._new_capture()
        if not capture.isOpened():
            capture.release()
            self._capture = None
            self._schedule_reconnect_backoff(now)
            LOGGER.warning(
                "Network video unavailable; next retry in %.0f seconds: %s",
                self._reconnect_delay,
                self.source,
            )
            return False
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        LOGGER.info("Reconnected network video stream: %s", self.source)
        return True

    def _schedule_reconnect_backoff(self, now: float | None = None) -> None:
        self._last_reconnect_attempt = time.perf_counter() if now is None else now
        self._reconnect_delay = min(
            max(self._reconnect_delay * 2.0, NETWORK_RECONNECT_INITIAL_DELAY_SECONDS),
            NETWORK_RECONNECT_MAX_DELAY_SECONDS,
        )

    def close(self) -> None:
        with self._lock:
            if self._ros_capture is not None:
                self._ros_capture.close()
                self._ros_capture = None
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._state = PlaybackState.STOPPED

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
