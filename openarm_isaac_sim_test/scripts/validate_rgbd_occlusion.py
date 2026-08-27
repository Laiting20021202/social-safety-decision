#!/usr/bin/env python3
"""Verify current-frame occlusion in 3D Safety's image-derived world cloud."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


PRODUCTION_WORLD_TOPIC = "/realtime_safety/environment_cloud_world"


class OcclusionValidator(Node):
    def __init__(self) -> None:
        super().__init__("validate_rgbd_occlusion")
        self.phase = "idle"
        self.counts: dict[str, list[int]] = {"clear": [], "blocked": []}
        self.received: dict[str, list[float]] = {"clear": [], "blocked": []}
        self.stamps: dict[str, list[int]] = {"clear": [], "blocked": []}
        self.production_frames: set[str] = set()
        self.sensor_diagnostic_frames = 0
        self.poses: dict[str, np.ndarray] = {}
        self._command = self.create_publisher(String, "/sim/hand/command", 10)
        self._target = self.create_publisher(
            PoseStamped, "/sim/hand/manual_target_pose", 10
        )
        self.create_subscription(
            PointCloud2,
            PRODUCTION_WORLD_TOPIC,
            self._cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/rgbd/points_world",
            self._sensor_diagnostic,
            qos_profile_sensor_data,
        )
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models, 10)

    def _models(self, message: ModelStates) -> None:
        self.poses = {
            name: np.asarray(
                [pose.position.x, pose.position.y, pose.position.z], dtype=float
            )
            for name, pose in zip(message.name, message.pose, strict=True)
            if name in {"rgbd_sensor", "left_target_cube"}
        }

    def _cloud(self, message: PointCloud2) -> None:
        if self.phase not in self.counts or "left_target_cube" not in self.poses:
            return
        self.received[self.phase].append(time.monotonic())
        stamp = message.header.stamp
        self.stamps[self.phase].append(
            int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        )
        self.production_frames.add(str(message.header.frame_id))
        offsets = {field.name: int(field.offset) for field in message.fields}
        if message.width == 0:
            self.counts[self.phase].append(0)
            return
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), int(message.row_step)
        )
        packed = np.ascontiguousarray(
            rows[:, : int(message.width) * int(message.point_step)]
        ).reshape(-1, int(message.point_step))

        def field(name: str) -> np.ndarray:
            offset = offsets[name]
            return np.ascontiguousarray(packed[:, offset : offset + 4]).view("<f4").reshape(-1)

        xyz = np.column_stack((field("x"), field("y"), field("z")))
        target = self.poses["left_target_cube"]
        points = int((np.abs(xyz - target) <= 0.04).all(axis=1).sum())
        self.counts[self.phase].append(points)

    def _sensor_diagnostic(self, _message: PointCloud2) -> None:
        # Kept only to make the report explicit: native Gazebo geometry does
        # not contribute to clear/blocked measurements or the success gate.
        self.sensor_diagnostic_frames += 1

    def command(self, value: str) -> None:
        self._command.publish(String(data=value))

    def occluding_pose(self) -> PoseStamped:
        camera = self.poses["rgbd_sensor"]
        target = self.poses["left_target_cube"]
        center = camera + 0.50 * (target - camera)
        message = PoseStamped()
        message.header.frame_id = "world"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(
            float, center
        )
        message.pose.orientation.w = 1.0
        return message

    def publish_target(self, message: PoseStamped) -> None:
        self._target.publish(message)


def spin(node: Node, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle-seconds", type=float, default=4.0)
    parser.add_argument("--sample-seconds", type=float, default=1.5)
    parser.add_argument("--maximum-remaining-ratio", type=float, default=0.15)
    args = parser.parse_args()
    rclpy.init()
    node = OcclusionValidator()
    try:
        node.command("withdraw")
        spin(node, args.settle_seconds)
        if not {"rgbd_sensor", "left_target_cube"}.issubset(node.poses):
            print(json.dumps({"passed": False, "reason": "model_poses_missing"}))
            return 1
        node.phase = "clear"
        spin(node, args.sample_seconds)
        node.phase = "idle"
        target = node.occluding_pose()
        for _ in range(3):
            node.publish_target(target)
            node.command("speed:0.6")
            node.command("manual:on")
            spin(node, 0.1)
        spin(node, args.settle_seconds)
        node.phase = "blocked"
        spin(node, args.sample_seconds)
        clear = float(np.median(node.counts["clear"])) if node.counts["clear"] else 0.0
        blocked = (
            float(np.median(node.counts["blocked"]))
            if node.counts["blocked"]
            else float("inf")
        )
        ratio = blocked / clear if clear > 0.0 else float("inf")
        rates = {
            phase: (
                (len(values) - 1) / max(values[-1] - values[0], 1e-9)
                if len(values) > 1
                else 0.0
            )
            for phase, values in node.received.items()
        }
        frames_are_world = node.production_frames == {"world"}
        stamp_unique_ratios = {
            phase: len(set(stamps)) / max(len(stamps), 1)
            for phase, stamps in node.stamps.items()
        }
        current_frames = (
            all(ratio >= 0.95 for ratio in stamp_unique_ratios.values())
            and len(set(node.stamps["clear"])) >= 10
            and len(set(node.stamps["blocked"])) >= 10
            and set(node.stamps["clear"]).isdisjoint(node.stamps["blocked"])
        )
        passed = (
            clear >= 10.0
            and ratio <= args.maximum_remaining_ratio
            and all(rate >= 10.0 for rate in rates.values())
            and frames_are_world
            and current_frames
        )
        report = {
            "passed": passed,
            "clear_cube_points_median": clear,
            "occluded_cube_points_median": blocked,
            "remaining_ratio": ratio,
            "clear_frames": len(node.counts["clear"]),
            "occluded_frames": len(node.counts["blocked"]),
            "rates_hz": rates,
            "frame_ids": sorted(node.production_frames),
            "unique_stamp_ratios": stamp_unique_ratios,
            "current_frames": current_frames,
            "source": PRODUCTION_WORLD_TOPIC,
            "source_contract": "3d_safety_decision_rgb_aligned_depth_camera_info_backprojection",
            "sensor_diagnostic_only": {
                "topic": "/rgbd/points_world",
                "frames": node.sensor_diagnostic_frames,
                "used_for_pass": False,
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if passed else 1
    finally:
        node.phase = "idle"
        node.command("withdraw")
        spin(node, 0.5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
