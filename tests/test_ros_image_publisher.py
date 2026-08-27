from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.ros2_bridge.image_publisher import ImageTopicPublisher
from realtime_safety.ros2_bridge.stamps import source_timestamp_or_now
from realtime_safety.types import FramePacket


class _Stamp:
    def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
        self.sec = sec
        self.nanosec = nanosec


class _Clock:
    def __init__(self, stamp: _Stamp) -> None:
        self._stamp = stamp

    def now(self) -> SimpleNamespace:
        return SimpleNamespace(
            to_msg=lambda: _Stamp(self._stamp.sec, self._stamp.nanosec)
        )


class _Image:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None, frame_id="")


class _CameraInfo:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None, frame_id="")


def _frame(timestamp: float) -> FramePacket:
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    return FramePacket(
        frame_index=4,
        source_timestamp=timestamp,
        capture_timestamp=10.0,
        bgr=bgr,
        rgb=bgr[..., ::-1],
        original_fps=30.0,
        original_width=32,
        original_height=24,
    )


def _publisher() -> tuple[ImageTopicPublisher, list[object], list[object]]:
    images: list[object] = []
    camera_infos: list[object] = []
    publisher = ImageTopicPublisher(
        frame_id="camera_frame",
        camera_info_topic="/camera/camera_info",
        focal_length_x=272.0,
        focal_length_y=273.0,
        principal_point_x=15.5,
        principal_point_y=11.5,
    )
    publisher._node = SimpleNamespace(get_clock=lambda: _Clock(_Stamp(99, 7)))
    publisher._publisher = SimpleNamespace(publish=images.append)
    publisher._image_type = _Image
    publisher._camera_info_publisher = SimpleNamespace(publish=camera_infos.append)
    publisher._camera_info_type = _CameraInfo
    return publisher, images, camera_infos


def test_image_and_camera_info_share_source_stamp_frame_and_intrinsics() -> None:
    publisher, images, camera_infos = _publisher()

    publisher.publish(_frame(123.25))

    assert len(images) == 1
    assert len(camera_infos) == 1
    image = images[0]
    camera_info = camera_infos[0]
    assert image.header.stamp.sec == 123
    assert image.header.stamp.nanosec == 250_000_000
    assert camera_info.header is image.header
    assert camera_info.header.frame_id == "camera_frame"
    assert (camera_info.width, camera_info.height) == (32, 24)
    assert camera_info.k == [
        272.0,
        0.0,
        15.5,
        0.0,
        273.0,
        11.5,
        0.0,
        0.0,
        1.0,
    ]
    assert camera_info.r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert camera_info.p == [
        272.0,
        0.0,
        15.5,
        0.0,
        0.0,
        273.0,
        11.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


@pytest.mark.parametrize("timestamp", [0.0, -1.0, float("nan"), float("inf")])
def test_image_invalid_source_stamp_falls_back_to_node_clock(timestamp: float) -> None:
    publisher, images, camera_infos = _publisher()

    publisher.publish(_frame(timestamp))

    assert (images[0].header.stamp.sec, images[0].header.stamp.nanosec) == (99, 7)
    assert camera_infos[0].header is images[0].header


def test_camera_info_configuration_requires_complete_valid_intrinsics() -> None:
    with pytest.raises(ValueError, match="requires fx, fy, cx, and cy"):
        ImageTopicPublisher(
            camera_info_topic="/camera/camera_info",
            focal_length_x=272.0,
        )
    with pytest.raises(ValueError, match="fx and fy must be positive"):
        ImageTopicPublisher(
            camera_info_topic="/camera/camera_info",
            focal_length_x=0.0,
            focal_length_y=272.0,
            principal_point_x=10.0,
            principal_point_y=10.0,
        )
    with pytest.raises(ValueError, match="camera_info_topic is required"):
        ImageTopicPublisher(
            focal_length_x=272.0,
            focal_length_y=272.0,
            principal_point_x=10.0,
            principal_point_y=10.0,
        )


def test_local_monotonic_source_stamp_is_translated_to_ros_clock(monkeypatch) -> None:
    monkeypatch.setattr(
        "realtime_safety.ros2_bridge.stamps.time.perf_counter",
        lambda: 100_000.0,
    )
    node = SimpleNamespace(
        get_clock=lambda: _Clock(_Stamp(1_700_000_000, 0))
    )

    stamp = source_timestamp_or_now(node, 99_999.5)

    assert (stamp.sec, stamp.nanosec) == (1_699_999_999, 500_000_000)


def test_unrelated_clock_domain_source_stamp_falls_back_to_node_clock(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "realtime_safety.ros2_bridge.stamps.time.perf_counter",
        lambda: 100_000.0,
    )
    node = SimpleNamespace(
        get_clock=lambda: _Clock(_Stamp(1_700_000_000, 123))
    )

    stamp = source_timestamp_or_now(node, 42.0)

    assert (stamp.sec, stamp.nanosec) == (1_700_000_000, 123)
