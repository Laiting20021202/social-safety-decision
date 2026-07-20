from pathlib import Path

import cv2
import numpy as np

from realtime_safety.pipeline import video_source as video_source_module
from realtime_safety.pipeline.video_source import CameraDevice, PlaybackState, VideoSource, discover_cameras


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
