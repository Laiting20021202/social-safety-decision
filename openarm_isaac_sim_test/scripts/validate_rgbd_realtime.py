#!/usr/bin/env python3
"""Verify 3D Safety regenerates PointCloud2 from every aligned depth frame."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


PRODUCTION_RAW_TOPIC = "/realtime_safety/pointcloud"
PRODUCTION_WORLD_TOPIC = "/realtime_safety/environment_cloud_world"


def _stamp(message: object) -> int:
    header = getattr(message, "header")
    value = getattr(header, "stamp")
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _rate(received: list[float]) -> float:
    if len(received) < 2:
        return 0.0
    return float((len(received) - 1) / max(received[-1] - received[0], 1e-9))


def _quaternion_rotation(pose: object) -> np.ndarray:
    q = getattr(pose, "orientation")
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _xyz(message: PointCloud2) -> np.ndarray:
    if int(message.width) * int(message.height) == 0:
        return np.empty((0, 3), dtype=np.float32)
    offsets = {field.name: int(field.offset) for field in message.fields}
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height), int(message.row_step)
    )
    packed = np.ascontiguousarray(
        rows[:, : int(message.width) * int(message.point_step)]
    ).reshape(-1, int(message.point_step))

    def field(name: str) -> np.ndarray:
        offset = offsets[name]
        return np.ascontiguousarray(packed[:, offset : offset + 4]).view("<f4").reshape(-1)

    return np.column_stack((field("x"), field("y"), field("z")))


class Validator(Node):
    def __init__(self, obstacle_topic: str) -> None:
        super().__init__("validate_rgbd_realtime")
        self.received: dict[str, list[float]] = defaultdict(list)
        self.stamps: dict[str, set[int]] = defaultdict(set)
        self.depth: dict[int, Image] = {}
        self.cloud: dict[int, PointCloud2] = {}
        self.cloud_samples: list[np.ndarray] = []
        self.info: dict[int, CameraInfo] = {}
        self.production_frames: dict[str, set[str]] = {
            "raw": set(),
            "world": set(),
        }
        self.sensor_diagnostic_counts = {"raw": 0, "world": 0}
        self.depth_error_m: float | None = None
        self.backprojection_error_m: float | None = None
        self.obstacle_points = 0
        self.obstacle_centroids: list[np.ndarray] = []
        self.obstacle_frame = ""
        self.hand_error_m: float | None = None
        self.poses: dict[str, object] = {}
        for name, message_type, topic, callback in (
            ("rgb", Image, "/rgbd/color/image_raw", self._rgb),
            ("depth", Image, "/rgbd/aligned_depth_to_color/image_raw", self._depth),
            ("raw", PointCloud2, PRODUCTION_RAW_TOPIC, self._raw),
            (
                "world",
                PointCloud2,
                PRODUCTION_WORLD_TOPIC,
                self._world,
            ),
            ("obstacle", PointCloud2, obstacle_topic, self._obstacle),
            ("info", CameraInfo, "/rgbd/color/camera_info", self._info),
        ):
            self.create_subscription(message_type, topic, callback, qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2,
            "/rgbd/points",
            lambda _: self._sensor_diagnostic("raw"),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/rgbd/points_world",
            lambda _: self._sensor_diagnostic("world"),
            qos_profile_sensor_data,
        )
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models, 10)

    def _record(self, name: str, message: object) -> None:
        self.received[name].append(time.monotonic())
        if hasattr(message, "header"):
            self.stamps[name].add(_stamp(message))

    def _rgb(self, message: Image) -> None:
        self._record("rgb", message)

    def _depth(self, message: Image) -> None:
        self._record("depth", message)
        self.depth[_stamp(message)] = message
        self._compare(_stamp(message))

    def _raw(self, message: PointCloud2) -> None:
        self._record("raw", message)
        self.production_frames["raw"].add(str(message.header.frame_id))
        self.cloud[_stamp(message)] = message
        points = _xyz(message)
        if len(points):
            self.cloud_samples.append(points)
            self.cloud_samples = self.cloud_samples[-30:]
        self._compare(_stamp(message))

    def _world(self, message: PointCloud2) -> None:
        self._record("world", message)
        self.production_frames["world"].add(str(message.header.frame_id))

    def _info(self, message: CameraInfo) -> None:
        self._record("info", message)
        stamp = _stamp(message)
        self.info[stamp] = message
        # CameraInfo can arrive after the image pair under DDS scheduling.
        self._compare(stamp)

    def _sensor_diagnostic(self, name: str) -> None:
        self.sensor_diagnostic_counts[name] += 1

    def _models(self, message: ModelStates) -> None:
        self.poses = dict(zip(message.name, message.pose, strict=True))

    def _obstacle(self, message: PointCloud2) -> None:
        self._record("obstacle", message)
        points = _xyz(message) if int(message.width) * int(message.height) else np.empty((0, 3))
        points = points[np.isfinite(points).all(axis=1)]
        self.obstacle_points = max(self.obstacle_points, len(points))
        self.obstacle_frame = str(message.header.frame_id)
        if len(points):
            self.obstacle_centroids.append(np.median(points, axis=0))
        hand = self.poses.get("human_hand")
        camera = self.poses.get("rgbd_sensor")
        if not len(points) or hand is None or camera is None:
            return
        if self.obstacle_frame == "world":
            world = points
        elif self.obstacle_frame == "rgbd_color_optical_frame":
            optical_to_link = np.asarray(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            )
            rotation = _quaternion_rotation(camera) @ optical_to_link
            translation = np.asarray(
                [camera.position.x, camera.position.y, camera.position.z]
            )
            world = points @ rotation.T + translation
        else:
            return
        hand_center = np.asarray([hand.position.x, hand.position.y, hand.position.z])
        # Segmentation covers fingers/palm, so compare the closest measured
        # point rather than assuming its centroid equals the model origin.
        error = float(np.min(np.linalg.norm(world - hand_center, axis=1)))
        self.hand_error_m = error if self.hand_error_m is None else min(self.hand_error_m, error)

    def _compare(self, stamp: int) -> None:
        if stamp not in self.depth or stamp not in self.cloud:
            return
        info = self.info.get(stamp)
        if info is None:
            # Static CameraInfo is permitted only when its stamp is explicitly
            # zero; never pair a random latest live calibration by accident.
            info = self.info.get(0)
        if info is None:
            return
        depth_message = self.depth[stamp]
        cloud_message = self.cloud[stamp]
        if depth_message.encoding != "32FC1":
            return
        rows = np.frombuffer(depth_message.data, dtype=np.uint8).reshape(
            int(depth_message.height), int(depth_message.step)
        )
        depth = np.ascontiguousarray(
            rows[:, : int(depth_message.width) * 4]
        ).view("<f4").reshape(int(depth_message.height), int(depth_message.width))
        xyz = _xyz(cloud_message)
        if not len(xyz):
            return
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
        finite_xyz = xyz[finite]
        if not len(finite_xyz):
            return
        z_cloud = finite_xyz[:, 2]
        u = np.rint(fx * finite_xyz[:, 0] / z_cloud + cx).astype(np.int64)
        v = np.rint(fy * finite_xyz[:, 1] / z_cloud + cy).astype(np.int64)
        inside = (
            (u >= 0)
            & (u < depth.shape[1])
            & (v >= 0)
            & (v < depth.shape[0])
        )
        u, v, sampled = u[inside], v[inside], finite_xyz[inside]
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            return
        expected_x = (u[valid] - cx) * z[valid] / fx
        expected_y = (v[valid] - cy) * z[valid] / fy
        self.depth_error_m = float(
            np.median(np.abs(sampled[:, 2][valid] - z[valid]))
        )
        self.backprojection_error_m = float(
            np.median(
                np.sqrt(
                    (sampled[:, 0][valid] - expected_x) ** 2
                    + (sampled[:, 1][valid] - expected_y) ** 2
                )
            )
        )

    def report(self) -> dict[str, object]:
        rgb = self.stamps["rgb"]
        depth = self.stamps["depth"]
        raw = self.stamps["raw"]
        world = self.stamps["world"]
        info = self.stamps["info"]
        denominator = max(len(raw), 1)
        obstacle_span = 0.0
        if len(self.obstacle_centroids) >= 2:
            centers = np.asarray(self.obstacle_centroids, dtype=np.float64)
            obstacle_span = float(np.linalg.norm(np.ptp(centers, axis=0)))
        temporal_delta = 0.0
        comparable = [
            (first, second)
            for first, second in zip(
                self.cloud_samples, self.cloud_samples[1:], strict=False
            )
            if first.shape == second.shape and len(first)
        ]
        if comparable:
            temporal_delta = float(
                np.median(
                    [
                        np.median(np.linalg.norm(second - first, axis=1))
                        for first, second in comparable
                    ]
                )
            )
        return {
            "rates_hz": {name: round(_rate(values), 3) for name, values in self.received.items()},
            "raw_cloud_dimensions": (
                [int(next(iter(self.cloud.values())).width), int(next(iter(self.cloud.values())).height)]
                if self.cloud else [0, 0]
            ),
            "rgb_depth_info_raw_exact_stamp_ratio": round(
                len(rgb & depth & info & raw) / denominator, 4
            ),
            "raw_world_exact_stamp_ratio": round(len(raw & world) / denominator, 4),
            "depth_z_median_error_m": self.depth_error_m,
            "backprojection_xy_median_error_m": self.backprojection_error_m,
            "cloud_temporal_median_delta_m": temporal_delta,
            "cloud_source": "3d_safety_rgb_aligned_depth_backprojection",
            "production_topics": {
                "raw": PRODUCTION_RAW_TOPIC,
                "world": PRODUCTION_WORLD_TOPIC,
            },
            "production_frame_ids": {
                name: sorted(values)
                for name, values in self.production_frames.items()
            },
            "sensor_diagnostic_only": {
                "/rgbd/points": self.sensor_diagnostic_counts["raw"],
                "/rgbd/points_world": self.sensor_diagnostic_counts["world"],
                "used_for_pass": False,
            },
            "obstacle_points": self.obstacle_points,
            "obstacle_centroid_span_m": obstacle_span,
            "obstacle_frame": self.obstacle_frame,
            "obstacle_to_hand_nearest_m_evaluator_only": self.hand_error_m,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--require-obstacle", action="store_true")
    parser.add_argument(
        "--obstacle-topic",
        default="/edgetam_tracker/obstacle_cloud_realtime",
    )
    parser.add_argument("--minimum-obstacle-rate", type=float, default=10.0)
    parser.add_argument("--minimum-obstacle-motion", type=float, default=0.0)
    args = parser.parse_args()
    rclpy.init()
    node = Validator(args.obstacle_topic)
    deadline = time.monotonic() + max(args.duration, 1.0)
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        report = node.report()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, indent=2, sort_keys=True))
    rates = report["rates_hz"]
    passed = all(
        float(rates.get(name, 0.0)) >= 10.0
        for name in ("rgb", "depth", "info", "raw", "world")
    )
    passed &= float(report["rgb_depth_info_raw_exact_stamp_ratio"]) >= 0.80
    passed &= float(report["raw_world_exact_stamp_ratio"]) >= 0.80
    passed &= report["raw_cloud_dimensions"] != [0, 0]
    passed &= report["production_frame_ids"]["raw"] == [
        "rgbd_color_optical_frame"
    ]
    passed &= report["production_frame_ids"]["world"] == ["world"]
    # The configurable 1.5 mm RGB-D noise is applied before backprojection.
    passed &= report["depth_z_median_error_m"] is not None and float(report["depth_z_median_error_m"]) < 4.5e-3
    passed &= report["backprojection_xy_median_error_m"] is not None and float(report["backprojection_xy_median_error_m"]) < 4.5e-3
    temporal_delta = float(report["cloud_temporal_median_delta_m"])
    passed &= 1e-5 < temporal_delta < 0.02
    if args.require_obstacle:
        obstacle_error = report["obstacle_to_hand_nearest_m_evaluator_only"]
        passed &= (
            int(report["obstacle_points"]) > 0
            and float(rates.get("obstacle", 0.0)) >= args.minimum_obstacle_rate
            and float(report["obstacle_centroid_span_m"])
            >= args.minimum_obstacle_motion
            and obstacle_error is not None
            and float(obstacle_error) < 0.15
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
