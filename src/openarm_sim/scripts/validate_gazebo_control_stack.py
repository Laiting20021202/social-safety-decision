#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class ControlStackProbe(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_control_stack_probe")
        self.diagnostics: dict[str, str] = {}
        self.task_state = ""
        self.safety_state = ""
        self.dynamic_status = ""
        self.perception_messages = 0
        self.create_subscription(
            DiagnosticArray, "/edgetam_tracker/diagnostics", self._diagnostics, 10
        )
        self.create_subscription(
            PointCloud2,
            "/perception/obstacles",
            self._perception,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/openarm/task/state", self._task_state, 10
        )
        self.create_subscription(
            String, "/openarm/safety/state", self._safety_state, 10
        )
        self.create_subscription(
            String,
            "/openarm/dynamic_avoidance/status",
            self._dynamic_status,
            10,
        )
        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.controllers = {
            side: ActionClient(
                self,
                FollowJointTrajectory,
                f"/{side}_joint_trajectory_controller/follow_joint_trajectory",
            )
            for side in ("left", "right")
        }

    def _diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            for value in status.values:
                self.diagnostics[value.key] = value.value

    def _perception(self, _message: PointCloud2) -> None:
        self.perception_messages += 1

    def _task_state(self, message: String) -> None:
        self.task_state = message.data

    def _safety_state(self, message: String) -> None:
        self.safety_state = message.data

    def _dynamic_status(self, message: String) -> None:
        self.dynamic_status = message.data

    def report(self) -> dict[str, object]:
        checks = {
            "edgetam_model": self.diagnostics.get("state") == "ready",
            "rgb_hand_model": self.diagnostics.get("hand_semantic_status")
            in {"ready", "tracking_held"},
            "perception_cloud_stream": self.perception_messages > 0,
            "move_group": self.move_group.server_is_ready(),
            "left_controller": self.controllers["left"].server_is_ready(),
            "right_controller": self.controllers["right"].server_is_ready(),
            "pose_goal": bool(self.task_state),
            "dynamic_avoidance": self.dynamic_status.startswith("READY "),
            "perception_isolation": "source=perception" in self.dynamic_status,
            "safety_supervisor": self.safety_state
            in {"SAFE", "WARNING", "PAUSE", "REPLAN", "RECOVER"},
        }
        return {
            "checks": checks,
            "missing": [name for name, passed in checks.items() if not passed],
            "diagnostics": {
                "edge_state": self.diagnostics.get("state", ""),
                "hand_semantic_status": self.diagnostics.get(
                    "hand_semantic_status", ""
                ),
                "background_state": self.diagnostics.get("background_state", ""),
            },
            "perception_messages": self.perception_messages,
            "task_state": self.task_state,
            "safety_state": self.safety_state,
            "dynamic_status": self.dynamic_status,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate live Gazebo control/model stack")
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = ControlStackProbe()
    deadline = time.monotonic() + max(args.seconds, 0.1)
    try:
        report = node.report()
        while report["missing"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            report = node.report()
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        return 0 if not report["missing"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
