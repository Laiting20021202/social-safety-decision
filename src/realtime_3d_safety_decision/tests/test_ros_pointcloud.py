import numpy as np
import pytest
from types import SimpleNamespace

from realtime_safety.ros2_bridge.pointcloud_publisher import (
    PointCloudTopicPublisher,
    _POINT_DTYPE,
    _pack_rgb_points,
)
from realtime_safety.types import PointCloudFrame


def test_pointcloud2_payload_packs_xyz_rgb_and_drops_nonfinite_points() -> None:
    points = np.array(((1.0, 2.0, 3.0), (np.nan, 0.0, 0.0), (-1.0, 0.5, 4.0)), dtype=np.float32)
    colors = np.array(((255, 128, 1), (9, 9, 9), (2, 3, 4)), dtype=np.uint8)
    data, count = _pack_rgb_points(points, colors)
    decoded = np.frombuffer(data, dtype=_POINT_DTYPE)

    assert count == 2
    np.testing.assert_allclose(decoded[["x", "y", "z"]][0].tolist(), (1.0, 2.0, 3.0))
    assert decoded["rgb"].tolist() == [0xFF8001, 0x020304]


def test_pointcloud2_payload_handles_empty_cloud() -> None:
    data, count = _pack_rgb_points(np.empty((0, 3)), np.empty((0, 3)))

    assert data == b""
    assert count == 0


def test_camera_y_forward_wire_mode_flips_internal_z_up_to_z_down() -> None:
    points = np.array(((0.2, 0.6, 0.15),), dtype=np.float32)
    colors = np.array(((255, 0, 0),), dtype=np.uint8)

    data, count = _pack_rgb_points(
        points,
        colors,
        coordinate_mode="camera_y_forward",
    )
    decoded = np.frombuffer(data, dtype=_POINT_DTYPE)

    assert count == 1
    np.testing.assert_allclose(
        decoded[["x", "y", "z"]][0].tolist(),
        (0.2, 0.6, -0.15),
    )
    # Packing must not mutate the GUI/internal z-up cloud.
    assert points[0, 2] == pytest.approx(0.15)


def test_ros_optical_wire_mode_uses_rep103_axes() -> None:
    points = np.array(((0.2, 0.6, 0.15),), dtype=np.float32)
    colors = np.array(((1, 2, 3),), dtype=np.uint8)

    data, count = _pack_rgb_points(
        points,
        colors,
        coordinate_mode="ros_optical",
    )
    decoded = np.frombuffer(data, dtype=_POINT_DTYPE)

    assert count == 1
    np.testing.assert_allclose(
        decoded[["x", "y", "z"]][0].tolist(),
        (0.2, -0.15, 0.6),
    )


def test_pointcloud_topic_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="rate must be positive"):
        PointCloudTopicPublisher(max_rate_hz=0)


def test_obstacle_publisher_emits_empty_cloud_to_clear_stale_obstacles() -> None:
    class FakePointCloud2:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None, frame_id="")

    class FakePointField:
        FLOAT32 = 7
        UINT32 = 6

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeClock:
        @staticmethod
        def now():
            return SimpleNamespace(to_msg=lambda: "stamp")

    messages = []
    publisher = PointCloudTopicPublisher(publish_empty=True)
    publisher._node = SimpleNamespace(get_clock=lambda: FakeClock())
    publisher._publisher = SimpleNamespace(publish=messages.append)
    publisher._pointcloud_type = FakePointCloud2
    publisher._pointfield_type = FakePointField
    cloud = PointCloudFrame(
        points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.uint8),
        confidence=np.empty((0,), dtype=np.float32),
        pointmap=np.empty((0, 0, 3), dtype=np.float32),
        frame_index=1,
        timestamp=0.0,
        anchor_frame_index=1,
        inference_ms=0.0,
        valid=False,
        source="test",
    )

    publisher.publish(cloud)

    assert len(messages) == 1
    assert messages[0].width == 0
    assert messages[0].row_step == 0
    assert messages[0].data == b""
    assert messages[0].header.stamp == "stamp"
    assert publisher.last_publish_duration_ms >= 0.0


def test_nonempty_message_has_consistent_pointcloud2_wire_layout() -> None:
    class FakePointCloud2:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None, frame_id="")

    class FakePointField:
        FLOAT32 = 7
        UINT32 = 6

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeClock:
        @staticmethod
        def now():
            return SimpleNamespace(to_msg=lambda: "stamp")

    messages = []
    publisher = PointCloudTopicPublisher(
        frame_id="realtime_safety_frame",
        coordinate_mode="camera_y_forward",
    )
    publisher._node = SimpleNamespace(get_clock=lambda: FakeClock())
    publisher._publisher = SimpleNamespace(publish=messages.append)
    publisher._pointcloud_type = FakePointCloud2
    publisher._pointfield_type = FakePointField
    cloud = PointCloudFrame(
        points=np.array(((0.1, 0.4, 0.2), (0.2, 0.5, -0.1)), dtype=np.float32),
        colors=np.array(((255, 0, 0), (0, 255, 0)), dtype=np.uint8),
        confidence=np.ones(2, dtype=np.float32),
        pointmap=np.empty((0, 0, 3), dtype=np.float32),
        frame_index=1,
        timestamp=0.0,
        anchor_frame_index=1,
        inference_ms=0.0,
        valid=True,
        source="test",
    )

    publisher.publish(cloud)

    message = messages[0]
    assert message.header.frame_id == "realtime_safety_frame"
    assert message.height == 1
    assert message.width == 2
    assert [(field.name, field.offset, field.datatype) for field in message.fields[:3]] == [
        ("x", 0, FakePointField.FLOAT32),
        ("y", 4, FakePointField.FLOAT32),
        ("z", 8, FakePointField.FLOAT32),
    ]
    assert message.point_step == 16
    assert message.row_step == 32
    assert len(message.data) == 32
    decoded = np.frombuffer(message.data, dtype=_POINT_DTYPE)
    np.testing.assert_allclose(decoded["z"], (-0.2, 0.1))


def test_pointcloud_header_prefers_valid_source_timestamp() -> None:
    class FakeStamp:
        def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
            self.sec = sec
            self.nanosec = nanosec

    class FakePointCloud2:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None, frame_id="")

    class FakePointField:
        FLOAT32 = 7
        UINT32 = 6

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeClock:
        @staticmethod
        def now():
            return SimpleNamespace(to_msg=lambda: FakeStamp(99, 7))

    messages = []
    publisher = PointCloudTopicPublisher()
    publisher._node = SimpleNamespace(get_clock=lambda: FakeClock())
    publisher._publisher = SimpleNamespace(publish=messages.append)
    publisher._pointcloud_type = FakePointCloud2
    publisher._pointfield_type = FakePointField
    cloud = PointCloudFrame(
        points=np.array(((0.1, 0.4, 0.2),), dtype=np.float32),
        colors=np.array(((255, 0, 0),), dtype=np.uint8),
        confidence=np.ones(1, dtype=np.float32),
        pointmap=np.empty((0, 0, 3), dtype=np.float32),
        frame_index=1,
        timestamp=456.75,
        anchor_frame_index=1,
        inference_ms=0.0,
        valid=True,
        source="test",
    )

    publisher.publish(cloud)

    stamp = messages[0].header.stamp
    assert (stamp.sec, stamp.nanosec) == (456, 750_000_000)


def test_pointcloud_header_preserves_trusted_sim_timestamp_across_clock_domains() -> None:
    class FakeStamp:
        def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
            self.sec = sec
            self.nanosec = nanosec

    class FakePointCloud2:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=None, frame_id="")

    class FakePointField:
        FLOAT32 = 7
        UINT32 = 6

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeClock:
        @staticmethod
        def now():
            return SimpleNamespace(to_msg=lambda: FakeStamp(1_787_203_000, 0))

    messages = []
    publisher = PointCloudTopicPublisher(preserve_source_timestamp=True)
    publisher._node = SimpleNamespace(get_clock=lambda: FakeClock())
    publisher._publisher = SimpleNamespace(publish=messages.append)
    publisher._pointcloud_type = FakePointCloud2
    publisher._pointfield_type = FakePointField
    cloud = PointCloudFrame(
        points=np.array(((0.1, 0.4, 0.2),), dtype=np.float32),
        colors=np.array(((255, 0, 0),), dtype=np.uint8),
        confidence=np.ones(1, dtype=np.float32),
        pointmap=np.empty((0, 0, 3), dtype=np.float32),
        frame_index=1,
        timestamp=10_444.625,
        anchor_frame_index=1,
        inference_ms=0.0,
        valid=True,
        source="synchronized_ros_rgbd",
    )

    publisher.publish(cloud)

    stamp = messages[0].header.stamp
    assert (stamp.sec, stamp.nanosec) == (10_444, 625_000_000)
