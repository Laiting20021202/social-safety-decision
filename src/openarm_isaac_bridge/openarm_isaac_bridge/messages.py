from __future__ import annotations

import struct
from typing import Any

import numpy as np


def stamp_from_seconds(seconds: float) -> Any:
    from builtin_interfaces.msg import Time

    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        whole += 1
        nanoseconds -= 1_000_000_000
    return Time(sec=whole, nanosec=nanoseconds)


def image_message(array: np.ndarray, encoding: str, frame_id: str, stamp: Any) -> Any:
    from sensor_msgs.msg import Image

    contiguous = np.ascontiguousarray(array)
    channels = 1 if contiguous.ndim == 2 else contiguous.shape[2]
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = int(contiguous.shape[0])
    message.width = int(contiguous.shape[1])
    message.encoding = encoding
    message.is_bigendian = False
    message.step = int(contiguous.shape[1] * channels * contiguous.dtype.itemsize)
    message.data = contiguous.tobytes()
    return message


def camera_info_message(intrinsics: Any, frame_id: str, stamp: Any) -> Any:
    from sensor_msgs.msg import CameraInfo

    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.width = intrinsics.width
    message.height = intrinsics.height
    message.distortion_model = "plumb_bob"
    message.d = [0.0] * 5
    message.k = [intrinsics.fx, 0.0, intrinsics.cx, 0.0, intrinsics.fy, intrinsics.cy, 0.0, 0.0, 1.0]
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [
        intrinsics.fx,
        0.0,
        intrinsics.cx,
        0.0,
        0.0,
        intrinsics.fy,
        intrinsics.cy,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return message


def point_cloud_message(points: np.ndarray, colors: np.ndarray, frame_id: str, stamp: Any) -> Any:
    from sensor_msgs.msg import PointCloud2, PointField

    if points.shape[0] != colors.shape[0]:
        raise ValueError("point/color counts differ")
    packed_rgb = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    cloud = np.empty(
        len(points),
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]),
    )
    cloud["x"], cloud["y"], cloud["z"] = points[:, 0], points[:, 1], points[:, 2]
    cloud["rgb"] = packed_rgb
    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = 1
    message.width = len(points)
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    message.is_bigendian = struct.pack("=I", 1)[0] == 0
    message.point_step = 16
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = cloud.tobytes()
    return message

