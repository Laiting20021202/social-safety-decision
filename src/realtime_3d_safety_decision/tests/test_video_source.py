from pathlib import Path

import cv2
import numpy as np
import pytest

from realtime_safety.pipeline import video_source as video_source_module
from realtime_safety.pipeline.video_source import CameraDevice, PlaybackState, VideoSource, discover_cameras
from realtime_safety.ros2_bridge.image_subscriber import normalize_camera_qos


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True, fps: float = 30.0) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 123_000.0
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def getBackendName(self) -> str:
        return "FAKE"

    def release(self) -> None:
        self.released = True


def test_file_timestamps_start_at_zero_and_follow_source_fps(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for value in range(3):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    source = VideoSource(str(path))
    source.open()
    frames = [source.read() for _ in range(3)]
    source.close()
    assert all(frame is not None for frame in frames)
    timestamps = [frame.source_timestamp for frame in frames if frame is not None]
    assert timestamps[0] == 0.0
    np.testing.assert_allclose(np.diff(timestamps), [0.1, 0.1], atol=0.02)


def test_camera_discovery_rejects_metadata_or_unreadable_nodes(monkeypatch) -> None:
    image = np.full((48, 64, 3), 80, dtype=np.uint8)
    captures = {
        0: FakeCapture([image]),
        1: FakeCapture([]),
        2: FakeCapture([], opened=False),
    }
    monkeypatch.setattr(video_source_module, "_candidate_camera_indices", lambda _max_index: [0, 1, 2])
    monkeypatch.setattr(video_source_module, "_open_camera_capture", lambda index: captures[index])
    monkeypatch.setattr(video_source_module, "_camera_name", lambda index: f"USB camera {index}")

    cameras = discover_cameras(max_index=3)

    assert [camera.index for camera in cameras] == [0]
    assert cameras[0].name == "USB camera 0"
    assert (cameras[0].width, cameras[0].height) == (64, 48)
    assert all(capture.released for capture in captures.values())


def test_auto_camera_alias_selects_first_readable_device(monkeypatch) -> None:
    camera = CameraDevice(4, "/dev/video4", "USB webcam", 1280, 720, 30.0, "V4L2")
    monkeypatch.setattr(video_source_module, "detect_camera", lambda: camera)

    source = VideoSource("auto")

    assert source.source == 4
    assert source.camera_info == camera
    assert VideoSource._normalize_source("/dev/video7") == 7


def test_webcam_frames_use_capture_time_and_rgb_conversion(monkeypatch) -> None:
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    capture = FakeCapture([bgr])
    monkeypatch.setattr(video_source_module, "_open_camera_capture", lambda _index: capture)
    monkeypatch.setattr(video_source_module, "_camera_name", lambda _index: "USB webcam")
    source = VideoSource(0)

    source.open()
    frame = source.read()
    source.stop()

    assert frame is not None
    assert frame.source_timestamp == frame.capture_timestamp
    assert frame.original_fps == 30.0
    assert frame.rgb[0, 0].tolist() == [0, 0, 255]
    assert capture.released
    assert source.state == PlaybackState.STOPPED


def test_http_mjpeg_is_live_and_reconnects_without_ending(monkeypatch) -> None:
    first = np.full((24, 32, 3), 10, dtype=np.uint8)
    second = np.full((24, 32, 3), 20, dtype=np.uint8)
    captures = [FakeCapture([first]), FakeCapture([second])]
    monkeypatch.setattr(video_source_module, "_open_network_capture", lambda _url: captures.pop(0))
    source = VideoSource("http://camera.local/stream?topic=/camera/image_raw&type=mjpeg")

    source.open()
    frame0 = source.read()
    disconnected = source.read()
    source._last_reconnect_attempt = 0.0
    frame1 = source.read()
    source.close()

    assert source.is_network_stream
    assert source.is_live
    assert disconnected is None
    assert frame0 is not None and frame1 is not None
    assert frame0.frame_index == 0
    assert frame1.frame_index == 1
    assert frame1.bgr[0, 0, 0] == 20


def test_unavailable_network_source_stays_running_and_reconnects(monkeypatch) -> None:
    frame = np.full((24, 32, 3), 30, dtype=np.uint8)
    captures = [FakeCapture([], opened=False), FakeCapture([frame])]
    monkeypatch.setattr(video_source_module, "_open_network_capture", lambda _url: captures.pop(0))
    source = VideoSource("http://camera.local/stream")

    source.open()
    assert source.state == PlaybackState.RUNNING
    assert not source.is_connected

    source._last_reconnect_attempt = 0.0
    recovered = source.read()
    source.close()

    assert recovered is not None
    assert recovered.bgr[0, 0, 0] == 30


def test_network_reconnect_uses_exponential_backoff(monkeypatch) -> None:
    frame = np.full((24, 32, 3), 40, dtype=np.uint8)
    captures = [FakeCapture([frame]), FakeCapture([], opened=False)]
    monkeypatch.setattr(video_source_module, "_open_network_capture", lambda _url: captures.pop(0))
    source = VideoSource("http://camera.local/stream")

    source.open()
    assert source.read() is not None
    assert source._reconnect_delay == 1.0

    assert source.read() is None
    assert source._reconnect_delay == 2.0
    source._last_reconnect_attempt = 0.0
    assert source.read() is None
    assert source._reconnect_delay == 4.0
    source.close()


def test_network_reconnect_backoff_is_capped_at_ten_seconds() -> None:
    source = VideoSource("http://camera.local/stream")
    source._reconnect_delay = 8.0

    source._schedule_reconnect_backoff(now=123.0)

    assert source._last_reconnect_attempt == 123.0
    assert source._reconnect_delay == 10.0


def test_ros2_image_topic_is_a_live_latest_frame_source(monkeypatch) -> None:
    bgr = np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8)

    class FakeRosCapture:
        def __init__(self) -> None:
            self.started = False
            self.closed = False
            self.samples = [(bgr, 123.5, 25.0)]

        @property
        def is_connected(self) -> bool:
            return self.started and not self.closed

        def start(self) -> None:
            self.started = True

        def read_latest(self):
            return self.samples.pop(0) if self.samples else None

        def close(self) -> None:
            self.closed = True

    capture = FakeRosCapture()
    monkeypatch.setattr(video_source_module, "_open_ros2_capture", lambda topic: capture)
    source = VideoSource("ros2:///custom/camera/image_raw")

    source.open()
    frame = source.read()
    duplicate = source.read()
    source.close()

    assert source.is_ros2_stream
    assert source.is_remote_stream
    assert source.is_live
    assert source.ros2_topic == "/custom/camera/image_raw"
    assert frame is not None
    assert frame.frame_index == 0
    assert frame.source_timestamp == 123.5
    assert frame.original_fps == 25.0
    np.testing.assert_array_equal(frame.bgr, bgr)
    assert duplicate is None
    assert capture.closed


def test_ros2_image_topic_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        VideoSource("ros2://relative/image")


def test_ros2_camera_qos_defaults_to_isaac_compatible_sensor_data() -> None:
    assert normalize_camera_qos(None) == "sensor_data"
    assert normalize_camera_qos("best-effort") == "best_effort"
    assert normalize_camera_qos("reliable") == "reliable"
    with pytest.raises(ValueError, match="Unsupported camera QoS"):
        normalize_camera_qos("transient_local")
