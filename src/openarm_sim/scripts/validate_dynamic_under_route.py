#!/usr/bin/env python3
"""Exercise the perception-only hand cloud and MoveIt under-route sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class RouteValidator(Node):
    def __init__(self) -> None:
        super().__init__("validate_dynamic_under_route")
        self.events: list[str] = []
        self.states: list[str] = []
        self.safety_states: list[str] = []
        self.maximum_obstacle_points = 0
        self.hand_target = self.create_publisher(
            PoseStamped, "/sim/hand/manual_target_pose", 10
        )
        self.robot_target = self.create_publisher(PoseStamped, "/openarm/target_pose", 10)
        self.hand_command = self.create_publisher(String, "/sim/hand/command", 10)
        self.task_command = self.create_publisher(String, "/openarm/task/command", 10)
        self.safety_command = self.create_publisher(
            String, "/openarm/safety/command", 10
        )
        self.planner_mode = self.create_publisher(String, "/openarm/planner/mode", 10)
        self.obstacle_source = self.create_publisher(
            String, "/openarm/obstacle_source", 10
        )
        self.create_subscription(String, "/openarm/events", self._event, 50)
        self.create_subscription(String, "/openarm/task/state", self._state, 10)
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(
            PointCloud2,
            "/perception/obstacles",
            self._obstacle,
            qos_profile_sensor_data,
        )

    def _event(self, message: String) -> None:
        self.events.append(message.data)

    def _state(self, message: String) -> None:
        if not self.states or message.data != self.states[-1]:
            self.states.append(message.data)

    def _safety(self, message: String) -> None:
        if not self.safety_states or message.data != self.safety_states[-1]:
            self.safety_states.append(message.data)

    def _obstacle(self, message: PointCloud2) -> None:
        self.maximum_obstacle_points = max(
            self.maximum_obstacle_points, int(message.width) * int(message.height)
        )

    def string(self, publisher: object, value: str) -> None:
        publisher.publish(String(data=value))


def spin(node: Node, seconds: float, predicate: object | None = None) -> bool:
    deadline = time.monotonic() + max(seconds, 0.0)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if callable(predicate) and bool(predicate()):
            return True
    return bool(predicate()) if callable(predicate) else True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=70.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = RouteValidator()
    report: dict[str, object] = {}
    try:
        node.string(node.obstacle_source, "perception")
        node.string(node.hand_command, "withdraw")
        spin(node, 4.0)
        node.string(node.safety_command, "reset")
        node.string(node.task_command, "reset")
        spin(node, 2.0)
        node.string(node.planner_mode, "dynamic")
        node.string(node.task_command, "home")
        spin(node, 20.0, lambda: bool(node.states and node.states[-1] == "HOME_REACHED"))

        hand = PoseStamped()
        hand.header.frame_id = "world"
        hand.pose.position.x = -0.30
        hand.pose.position.y = 0.18
        # This pose is consistently visible to the real hand model and blocks
        # the descending home-to-goal segment with a feasible gap below.
        hand.pose.position.z = 0.43
        hand.pose.orientation.w = 1.0
        for _ in range(3):
            node.hand_target.publish(hand)
            node.string(node.hand_command, "speed:0.6")
            node.string(node.hand_command, "manual:on")
            spin(node, 0.1)
        # The parked-to-workspace travel is about 1.6 m.  Do not plan against
        # an intermediate segmentation box while the rate-limited hand is
        # still entering the camera frustum.
        spin(node, 4.5)
        node.maximum_obstacle_points = 0
        detected = spin(
            node,
            20.0,
            lambda: node.maximum_obstacle_points >= 100,
        )
        goal = PoseStamped()
        goal.header.frame_id = "world"
        # Stay just beyond the perceived hand box.  This is a reachable
        # post-obstacle target for the left arm and still forces a crossing.
        goal.pose.position.x = -0.12
        goal.pose.position.y = 0.18
        goal.pose.position.z = 0.34
        goal.pose.orientation.w = 1.0
        for _ in range(3):
            node.robot_target.publish(goal)
            spin(node, 0.1)
        if detected:
            node.string(node.task_command, "move_target")
        withdrew_after_entry = False
        deadline = time.monotonic() + max(args.timeout, 0.0)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            entry_seen = any(
                value == "dynamic_under_waypoint_reached,left,under_entry"
                for value in node.events
            )
            if entry_seen and not withdrew_after_entry:
                # Reproduce the GUI workflow: once the arm has visibly taken
                # the lower bypass, pull the hand away and verify recovery.
                node.string(node.hand_command, "withdraw")
                withdrew_after_entry = True
            if node.states and (
                node.states[-1] == "FINGERTIP_TARGETS_REACHED"
                or node.states[-1].startswith("PLANNING_FAILED")
                or node.states[-1] == "EMERGENCY_STOP"
            ):
                break
        completed = bool(
            node.states and node.states[-1] == "FINGERTIP_TARGETS_REACHED"
        )
        route_events = [
            value
            for value in node.events
            if value.startswith("dynamic_under_route_selected,")
        ]
        entry_reached = any(
            value == "dynamic_under_waypoint_reached,left,under_entry"
            for value in node.events
        )
        exit_reached = any(
            value == "dynamic_under_waypoint_reached,left,under_exit"
            for value in node.events
        )
        passed = bool(
            detected
            and route_events
            and entry_reached
            and exit_reached
            and completed
            and "EMERGENCY_STOP" not in node.safety_states
        )
        report = {
            "passed": passed,
            "obstacle_source": "perception",
            "maximum_perception_obstacle_points": node.maximum_obstacle_points,
            "route_events": route_events,
            "entry_reached": entry_reached,
            "exit_reached": exit_reached,
            "withdrew_after_entry": withdrew_after_entry,
            "target_reached": completed,
            "task_states": node.states,
            "safety_states": node.safety_states,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0 if passed else 1
    finally:
        node.string(node.hand_command, "withdraw")
        spin(node, 4.0)
        node.string(node.safety_command, "reset")
        node.string(node.obstacle_source, "perception")
        node.string(node.task_command, "home")
        spin(node, 20.0, lambda: bool(node.states and node.states[-1] == "HOME_REACHED"))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
