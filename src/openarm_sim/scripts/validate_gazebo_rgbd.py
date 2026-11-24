#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2


PRODUCTION_RAW_TOPIC = "/realtime_safety/pointcloud"
PRODUCTION_WORLD_TOPIC = "/realtime_safety/environment_cloud_world"


class Collector(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_phase1_validator")
        self.counts: dict[str, int] = defaultdict(int)
        self.first: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.messages: dict[str, object] = {}
        subscriptions = (
            ("clock", Clock, "/clock"),
            ("joints", JointState, "/joint_states"),
            ("rgb", Image, "/rgbd/color/image_raw"),
            ("depth", Image, "/rgbd/depth/image_raw"),
            ("aligned", Image, "/rgbd/aligned_depth_to_color/image_raw"),
            ("camera_info", CameraInfo, "/rgbd/color/camera_info"),
            ("production_raw_cloud", PointCloud2, PRODUCTION_RAW_TOPIC),
            ("production_world_cloud", PointCloud2, PRODUCTION_WORLD_TOPIC),
            # Gazebo's native clouds are sensor diagnostics only. They cannot
            # satisfy the production readiness gate because they bypass the
            # 3d_safety_decision RGB + aligned-depth backprojection.
            ("sensor_raw_cloud_diagnostic", PointCloud2, "/rgbd/points"),
            ("sensor_world_cloud_diagnostic", PointCloud2, "/rgbd/points_world"),
            ("gui_rgb", Image, "/realtime_safety/camera/image_raw"),
        )
        for key, message_type, topic in subscriptions:
            self.create_subscription(
                message_type,
                topic,
                lambda message, name=key: self._receive(name, message),
                qos_profile_sensor_data,
            )

    def _receive(self, name: str, message: object) -> None:
        now = time.monotonic()
        self.counts[name] += 1
        self.first.setdefault(name, now)
        self.last[name] = now
        self.messages[name] = message


def _points(message: PointCloud2) -> np.ndarray:
    if int(message.width) * int(message.height) == 0:
        return np.empty((0, 3), dtype=np.float32)
    offsets = {field.name: field.offset for field in message.fields}
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.row_step)
    data = np.ascontiguousarray(rows[:, : message.width * message.point_step]).reshape(
        -1, message.point_step
    )
    values = []
    for name in ("x", "y", "z"):
        offset = offsets[name]
        values.append(np.ascontiguousarray(data[:, offset : offset + 4]).view("<f4").reshape(-1))
    xyz = np.column_stack(values)
    return xyz[np.isfinite(xyz).all(axis=1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gui-preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Collector()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    rgb = node.messages.get("rgb")
    # The evidence image must be the exact aligned stream used by the
    # production RGB-D backprojector, not the native unaligned depth stream.
    depth = node.messages.get("aligned")
    raw_cloud = node.messages.get("production_raw_cloud")
    world_cloud = node.messages.get("production_world_cloud")
    required = {
        "clock",
        "joints",
        "rgb",
        "depth",
        "aligned",
        "camera_info",
        "production_raw_cloud",
        "production_world_cloud",
    }
    if args.require_gui_preview:
        required.add("gui_rgb")
    missing = sorted(required - node.messages.keys())
    report: dict[str, object] = {
        "missing": missing,
        "counts": dict(node.counts),
        "production_clouds": {
            "raw": PRODUCTION_RAW_TOPIC,
            "world": PRODUCTION_WORLD_TOPIC,
            "contract": "3d_safety_decision_rgb_aligned_depth_camera_info_backprojection",
        },
        "sensor_diagnostic_only": ["/rgbd/points", "/rgbd/points_world"],
    }
    report["rates_hz"] = {
        name: (node.counts[name] - 1) / (node.last[name] - node.first[name])
        if node.counts[name] > 1
        else 0.0
        for name in node.counts
    }
    if isinstance(rgb, Image):
        channels = 3 if rgb.encoding.lower() in {"rgb8", "bgr8"} else 4
        image = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.step)[:, : rgb.width * channels]
        image = image.reshape(rgb.height, rgb.width, channels)
        if rgb.encoding.lower() == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.output / "rgb.png"), image)
        report["rgb"] = {
            "encoding": rgb.encoding,
            "shape": [rgb.height, rgb.width],
            "frame_id": rgb.header.frame_id,
            "stamp_ns": rgb.header.stamp.sec * 1_000_000_000 + rgb.header.stamp.nanosec,
        }
    if isinstance(depth, Image):
        depth_m = np.frombuffer(depth.data, dtype="<f4").reshape(depth.height, depth.step // 4)[:, : depth.width]
        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        lo, hi = (np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0))
        depth_u8 = np.clip((depth_m - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.output / "depth.png"), cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))
        report["depth"] = {
            "topic": "/rgbd/aligned_depth_to_color/image_raw",
            "encoding": depth.encoding,
            "frame_id": depth.header.frame_id,
            "minimum_m": float(valid.min()) if valid.size else None,
            "median_m": float(np.median(valid)) if valid.size else None,
            "maximum_m": float(valid.max()) if valid.size else None,
        }
    for name, message in (
        ("production_raw_cloud", raw_cloud),
        ("production_world_cloud", world_cloud),
    ):
        if isinstance(message, PointCloud2):
            xyz = _points(message)
            report[name] = {
                "frame_id": message.header.frame_id,
                "points": int(xyz.shape[0]),
                "minimum_xyz": xyz.min(axis=0).tolist() if xyz.size else None,
                "median_xyz": np.median(xyz, axis=0).tolist() if xyz.size else None,
                "maximum_xyz": xyz.max(axis=0).tolist() if xyz.size else None,
            }
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    rates = report["rates_hz"]
    production_rates_ok = all(
        float(rates.get(name, 0.0)) >= 10.0
        for name in (
            "rgb",
            "aligned",
            "camera_info",
            "production_raw_cloud",
            "production_world_cloud",
        )
    )
    production_clouds_nonempty = all(
        isinstance(report.get(name), dict)
        and int(report[name].get("points", 0)) > 0
        for name in ("production_raw_cloud", "production_world_cloud")
    )
    production_frames_ok = (
        report.get("production_raw_cloud", {}).get("frame_id")
        == "rgbd_color_optical_frame"
        and report.get("production_world_cloud", {}).get("frame_id") == "world"
    )
    report["production_gate"] = {
        "minimum_rate_hz": 10.0,
        "rate_topics": [
            "/rgbd/color/image_raw",
            "/rgbd/aligned_depth_to_color/image_raw",
            "/rgbd/color/camera_info",
            PRODUCTION_RAW_TOPIC,
            PRODUCTION_WORLD_TOPIC,
        ],
        "rates_ok": production_rates_ok,
        "nonempty": production_clouds_nonempty,
        "frames_ok": production_frames_ok,
        "passed": not missing
        and production_rates_ok
        and production_clouds_nonempty
        and production_frames_ok,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["production_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
