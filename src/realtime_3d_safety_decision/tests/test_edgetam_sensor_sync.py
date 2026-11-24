from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.edgetam_tracker.sensor_sync import (
    CameraIntrinsics,
    RosSensorSynchronizer,
    camera_intrinsics_from_info,
    depth_message_to_meters,
    depth_to_cloud,
    pointcloud2_to_cloud,
    transform_points,
    validate_timestamps,
)


def _header(stamp: float = 1.0, frame: str = "camera") -> SimpleNamespace:
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame
    )


def test_timestamp_validation_rejects_stale_and_unsynchronized_data() -> None:
    assert validate_timestamps([1.00, 1.02], slop_sec=0.03).valid
    mismatch = validate_timestamps([1.00, 1.10], slop_sec=0.03)
    assert not mismatch.valid and mismatch.reason == "timestamps_out_of_sync"
    stale = validate_timestamps(
        [1.0, 1.01], slop_sec=0.03, now_sec=2.0, max_data_age_sec=0.2
    )
    assert not stale.valid and stale.reason == "sensor_data_stale"
    partially_stale = validate_timestamps(
        [1.0, 1.3],
        slop_sec=0.5,
        now_sec=1.4,
        max_data_age_sec=0.2,
    )
    assert (
        not partially_stale.valid
        and partially_stale.reason == "sensor_data_stale"
    )


def test_invalid_camera_info_is_not_silently_accepted() -> None:
    info = SimpleNamespace(
        k=[0.0] * 9, width=640, height=480, header=_header()
    )
    with pytest.raises(ValueError, match="fx/fy"):
        camera_intrinsics_from_info(info)


def test_camera_info_buffer_matches_reference_timestamp_not_latest() -> None:
    synchronizer = object.__new__(RosSensorSynchronizer)
    synchronizer.slop_sec = 0.05
    older = SimpleNamespace(header=_header(10.00))
    matching = SimpleNamespace(header=_header(10.10))
    newer = SimpleNamespace(header=_header(10.20))
    synchronizer._latest = {"camera_info": newer}
    synchronizer._camera_info_by_stamp = OrderedDict(
        ((10.00, older), (10.10, matching), (10.20, newer))
    )
    image = SimpleNamespace(header=_header(10.11))

    assert synchronizer._matching_camera_info(image) is matching


def test_depth_conversion_filters_invalid_values_and_preserves_pixels() -> None:
    raw_mm = np.array([[1000, 0], [2000, 3000]], dtype="<u2")
    message = SimpleNamespace(
        height=2,
        width=2,
        step=4,
        encoding="16UC1",
        is_bigendian=False,
        data=raw_mm.tobytes(),
    )
    depth = depth_message_to_meters(message)
    assert np.isnan(depth[0, 1])
    intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 2, 2, "camera")
    cloud = depth_to_cloud(depth, intrinsics, stamp=1.0)
    assert cloud.points.shape == (3, 3)
    np.testing.assert_allclose(cloud.points[0], [0.0, 0.0, 1.0])
    assert tuple(cloud.pixels_uv[-1]) == (1, 1)


def test_pointcloud2_parser_drops_nonfinite_and_handles_row_padding() -> None:
    point_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
    )
    points = np.zeros((2, 2), dtype=point_dtype)
    points["x"] = [[1.0, np.nan], [3.0, 4.0]]
    points["y"] = 2.0
    points["z"] = 3.0
    points["rgb"] = 0x112233
    padded = bytearray(2 * 40)
    for row in range(2):
        padded[row * 40 : row * 40 + 32] = points[row].tobytes()
    fields = [
        SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)
        for name, offset, datatype in (
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("rgb", 12, 6),
        )
    ]
    message = SimpleNamespace(
        header=_header(2.5, "depth_optical"),
        height=2,
        width=2,
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=40,
        data=bytes(padded),
    )
    cloud = pointcloud2_to_cloud(message)
    assert cloud.points.shape == (3, 3)
    assert cloud.image_shape == (2, 2)
    assert tuple(cloud.colors[0]) == (0x11, 0x22, 0x33)
    assert tuple(cloud.pixels_uv[-1]) == (1, 1)
    assert cloud.stamp == pytest.approx(2.5)


def test_pointcloud2_preserves_current_empty_frame_and_rejects_misalignment() -> None:
    fields = [
        SimpleNamespace(name=name, offset=offset, datatype=7, count=1)
        for name, offset in (("x", 0), ("y", 4), ("z", 8))
    ]
    empty = SimpleNamespace(
        header=_header(),
        height=1,
        width=0,
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=0,
        data=b"",
    )
    empty_cloud = pointcloud2_to_cloud(empty)
    assert empty_cloud.points.shape == (0, 3)
    assert empty_cloud.source_indices.shape == (0,)

    organized = SimpleNamespace(
        header=_header(),
        height=2,
        width=2,
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=24,
        data=np.zeros((2, 2, 3), dtype="<f4").tobytes(),
    )
    with pytest.raises(ValueError, match="image shape mismatch"):
        pointcloud2_to_cloud(organized, image_shape=(1, 4))


def test_transform_points_uses_real_rigid_transform() -> None:
    result = transform_points(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
    )
    np.testing.assert_allclose(result[0], [0.0, 3.0, 0.0], atol=1e-6)
