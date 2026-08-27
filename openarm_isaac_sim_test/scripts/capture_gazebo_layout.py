#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import rclpy
import yaml
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node


DEFAULT_MODELS = (
    "rgbd_sensor",
    "work_table",
    "apriltag_36h11_0",
    "left_target_cube",
    "right_target_cube",
)


def _clean(value: float, epsilon: float = 1e-8) -> float:
    return 0.0 if abs(value) < epsilon else float(value)


def _rpy_degrees(x: float, y: float, z: float, w: float) -> list[float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [_clean(math.degrees(value)) for value in (roll, pitch, yaw)]


class PoseCapture(Node):
    def __init__(self) -> None:
        super().__init__("capture_gazebo_layout")
        self.message: ModelStates | None = None
        self.create_subscription(ModelStates, "/gazebo/model_states", self._states, 10)

    def _states(self, message: ModelStates) -> None:
        self.message = message


def main() -> int:
    project = Path(
        os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    parser = argparse.ArgumentParser(
        description="Persist direct Gazebo camera/table/item model edits"
    )
    parser.add_argument(
        "--output", type=Path, default=project / "config/gazebo_layout.yaml"
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    rclpy.init()
    node = PoseCapture()
    deadline = time.monotonic() + max(args.timeout, 0.1)
    try:
        while node.message is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.message is None:
            raise RuntimeError("no /gazebo/model_states received")
        by_name = dict(zip(node.message.name, node.message.pose, strict=True))
        missing = sorted(set(args.models) - set(by_name))
        if missing:
            raise RuntimeError(f"Gazebo models not found: {', '.join(missing)}")
        models = {}
        for name in args.models:
            pose = by_name[name]
            models[name] = {
                "position": [
                    _clean(pose.position.x),
                    _clean(pose.position.y),
                    _clean(pose.position.z),
                ],
                "rpy_deg": _rpy_degrees(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
            }
        document = {"schema_version": 1, "models": models}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(document, sort_keys=True))
        temporary.replace(args.output)
        print(f"saved {len(models)} Gazebo model poses to {args.output}")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
