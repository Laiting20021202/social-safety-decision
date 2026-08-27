#!/usr/bin/env python3
"""Execute the high-level dual-target command and verify both TCP endpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformListener


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("validate_dual_target_motion")
        self.state = ""
        self.safety = ""
        self.transitions: list[str] = []
        self.safety_transitions: list[str] = []
        self.events: list[str] = []
        self.minimum_distance = float("inf")
        self.ground_truth_minimum_distance = float("inf")
        self.models: dict[str, object] = {}
        self.command = self.create_publisher(String, "/openarm/task/command", 10)
        self.hand_command = self.create_publisher(String, "/sim/hand/command", 10)
        self.hand_target = self.create_publisher(
            PoseStamped, "/sim/hand/manual_target_pose", 10
        )
        self.create_subscription(String, "/openarm/task/state", self._state, 10)
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(String, "/openarm/events", self._event, 50)
        self.create_subscription(
            Float64, "/openarm/safety/min_distance", self._distance, 10
        )
        self.create_subscription(
            Float64,
            "/sim/ground_truth/min_distance",
            self._ground_truth_distance,
            10,
        )
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models, 10)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def _state(self, message: String) -> None:
        if message.data != self.state:
            self.state = message.data
            self.transitions.append(self.state)

    def _safety(self, message: String) -> None:
        if message.data != self.safety:
            self.safety_transitions.append(message.data)
        self.safety = message.data

    def _event(self, message: String) -> None:
        self.events.append(message.data)

    def _distance(self, message: Float64) -> None:
        if np.isfinite(message.data):
            self.minimum_distance = min(self.minimum_distance, float(message.data))

    def _ground_truth_distance(self, message: Float64) -> None:
        if np.isfinite(message.data):
            self.ground_truth_minimum_distance = min(
                self.ground_truth_minimum_distance, float(message.data)
            )

    def _models(self, message: ModelStates) -> None:
        self.models = dict(zip(message.name, message.pose, strict=True))

    def errors(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for side in ("left", "right"):
            target = self.models.get(f"{side}_target_cube")
            if target is None:
                result[side] = None
                continue
            try:
                transform = self.tf.lookup_transform(
                    "world",
                    f"openarm_{side}_hand_tcp",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.25),
                )
            except Exception:
                result[side] = None
                continue
            actual = transform.transform.translation
            result[side] = float(
                np.linalg.norm(
                    [
                        actual.x - target.position.x,
                        actual.y - target.position.y,
                        actual.z - target.position.z,
                    ]
                )
            )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=70.0)
    parser.add_argument(
        "--post-target-avoidance",
        action="store_true",
        help=(
            "after both TCPs are holding their targets, start the fixed-height "
            "hand sweep and require evade/restore without commanding Home"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=0.045)
    parser.add_argument(
        "--sweep-park-y",
        type=float,
        default=-1.28,
        help=(
            "stage the hand outside both TCP goals at this world-Y before "
            "commanding the arms; X/Z and orientation remain unchanged"
        ),
    )
    parser.add_argument(
        "--sweep-approach-y",
        type=float,
        default=-0.40,
        help=(
            "after both targets are held, approach this still-clear world-Y "
            "position quickly before starting the slow equal-speed sweep"
        ),
    )
    parser.add_argument(
        "--sweep-x",
        type=float,
        default=-0.0820019543,
        help="calibrated world-X for the perception interaction sweep",
    )
    parser.add_argument(
        "--sweep-z",
        type=float,
        default=0.2618689835,
        help="calibrated fixed world-Z for the perception interaction sweep",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    startup = time.monotonic() + 8.0
    while time.monotonic() < startup and (
        not node.state or not node.safety or len(node.models) == 0
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
    if args.post_target_avoidance:
        # The sweep endpoints overlap the two marker lanes by design.  Starting
        # a target request with the hand already parked on an endpoint tests a
        # different problem (active-motion routing) and prevents the requested
        # target-hold -> lift -> restore sequence from ever starting.  Stage it
        # along world Y only, retaining the operator-calibrated X/Z/orientation.
        hand = node.models.get("human_hand")
        if hand is None:
            raise RuntimeError("Gazebo human_hand model is unavailable")
        staged = PoseStamped()
        staged.header.frame_id = "world"
        staged.header.stamp = node.get_clock().now().to_msg()
        staged.pose = hand
        staged.pose.position.x = float(args.sweep_x)
        staged.pose.position.y = float(args.sweep_park_y)
        staged.pose.position.z = float(args.sweep_z)
        node.hand_command.publish(String(data="auto_sweep:off"))
        node.hand_command.publish(String(data="speed:0.1"))
        node.hand_target.publish(staged)
        rclpy.spin_once(node, timeout_sec=0.25)
        node.hand_command.publish(String(data="manual:on"))
        stage_deadline = time.monotonic() + 12.0
        while time.monotonic() < stage_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            actual = node.models.get("human_hand")
            if actual is not None and abs(
                float(actual.position.y) - float(args.sweep_park_y)
            ) <= 0.015:
                break
        else:
            raise RuntimeError("hand did not reach the pre-target sweep park pose")
        # Keep manual mode engaged at the park target until both robot arms
        # have arrived.  Switching it off here lets the controller resync to
        # an in-flight Gazebo pose and the obstacle can drift back into the
        # target lane before the test has actually started.
        settle_deadline = time.monotonic() + 3.0
        while time.monotonic() < settle_deadline:
            node.hand_target.publish(staged)
            rclpy.spin_once(node, timeout_sec=0.1)
    # Make the probe repeatable even if a previous manual GUI request is still
    # executing; do not interpret a latched terminal state as this run.
    node.command.publish(String(data="reset"))
    reset_deadline = time.monotonic() + 4.0
    while time.monotonic() < reset_deadline and node.state != "IDLE":
        rclpy.spin_once(node, timeout_sec=0.1)
    node.command.publish(String(data="home"))
    home_deadline = time.monotonic() + 45.0
    while time.monotonic() < home_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.state == "HOME_REACHED":
            break
        if any(
            token in node.state
            for token in ("FAILED", "REJECTED", "NOT_READY", "UNREACHABLE")
        ):
            raise RuntimeError(f"pre-test HOME failed: {node.state}")
    else:
        raise RuntimeError(f"pre-test HOME timed out: {node.state}")
    node.transitions.clear()
    node.events.clear()
    node.command.publish(String(data="move_both_targets"))
    deadline = time.monotonic() + max(args.timeout, 1.0)
    target_holding_reached = False
    saw_active = False
    minimum_errors: dict[str, float] = {
        "left": float("inf"),
        "right": float("inf"),
    }
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        for side, value in node.errors().items():
            if value is not None:
                minimum_errors[side] = min(minimum_errors[side], value)
        if node.state in {
            "PLANNING",
            "DYNAMIC_SAFETY_CHECK",
            "EXECUTING",
            "LEFT_TARGET_REACHED",
            "RIGHT_TARGET_REACHED",
        }:
            saw_active = True
        motion_complete = any(
            event.startswith("motion_complete,both_targets,")
            for event in node.events
        )
        if saw_active and (
            node.state
            in {
                "FINGERTIP_TARGETS_REACHED",
                "TARGETS_REACHED_HOLDING",
            }
            or motion_complete
        ):
            target_holding_reached = True
            break
        if any(
            token in node.state
            for token in ("FAILED", "REJECTED", "NOT_READY", "UNREACHABLE")
        ):
            break
        if node.state == "EMERGENCY_STOP" or node.safety == "EMERGENCY_STOP":
            break
    target_holding_errors = node.errors()
    avoidance_hold_seen = False
    target_restore_seen = False
    if target_holding_reached and args.post_target_avoidance:
        # The obstacle must not chase either target. It continues along the
        # calibrated world-Y sweep at its current X/Z.  First return from the
        # off-workspace parking point to a still-clear approach point, then
        # enter the interaction region at the requested slow sweep speed.
        hand = node.models.get("human_hand")
        if hand is None:
            raise RuntimeError("Gazebo human_hand model disappeared")
        approach = PoseStamped()
        approach.header.frame_id = "world"
        approach.header.stamp = node.get_clock().now().to_msg()
        approach.pose = hand
        approach.pose.position.y = float(args.sweep_approach_y)
        node.hand_command.publish(String(data="speed:0.1"))
        node.hand_target.publish(approach)
        rclpy.spin_once(node, timeout_sec=0.25)
        node.hand_command.publish(String(data="manual:on"))
        approach_deadline = time.monotonic() + 15.0
        while time.monotonic() < approach_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            actual = node.models.get("human_hand")
            if actual is not None and abs(
                float(actual.position.y) - float(args.sweep_approach_y)
            ) <= 0.015:
                break
        else:
            raise RuntimeError("hand did not reach the pre-sweep approach pose")
        node.hand_command.publish(String(data="speed:0.01"))
        node.hand_command.publish(String(data="auto_sweep:on"))
        avoidance_deadline = time.monotonic() + max(args.timeout, 1.0)
        while time.monotonic() < avoidance_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.state == "IDLE_AVOIDANCE_HOLD":
                avoidance_hold_seen = True
            if avoidance_hold_seen and node.state == "TARGETS_RESTORED_HOLDING":
                target_restore_seen = True
                break
            if node.state == "EMERGENCY_STOP" or node.safety == "EMERGENCY_STOP":
                break
            if any(token in node.state for token in ("FAILED", "REJECTED")):
                break
    elif target_holding_reached:
        target_restore_seen = True
    final_errors = node.errors()
    measured_minimum_errors = {
        side: (value if np.isfinite(value) else None)
        for side, value in minimum_errors.items()
    }
    report = {
        "terminal_state": node.state,
        "safety_state": node.safety,
        "transitions": node.transitions,
        "safety_transitions": node.safety_transitions,
        "minimum_safety_distance_m": (
            node.minimum_distance if np.isfinite(node.minimum_distance) else None
        ),
        "ground_truth_minimum_distance_m_evaluator_only": (
            node.ground_truth_minimum_distance
            if np.isfinite(node.ground_truth_minimum_distance)
            else None
        ),
        "dynamic_events": [
            event
            for event in node.events
            if event.startswith(("dynamic_", "safety_transition,"))
        ],
        "completion_events": [
            event for event in node.events if event.startswith("motion_complete,")
        ],
        "target_holding_reached": target_holding_reached,
        "target_holding_error_m": target_holding_errors,
        "post_target_avoidance_requested": args.post_target_avoidance,
        "avoidance_hold_seen": avoidance_hold_seen,
        "target_restore_seen": target_restore_seen,
        "minimum_tcp_target_error_m": measured_minimum_errors,
        "final_tcp_target_error_m": final_errors,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    node.destroy_node()
    rclpy.shutdown()
    ground_truth_clear = (
        report["ground_truth_minimum_distance_m_evaluator_only"] is not None
        and float(report["ground_truth_minimum_distance_m_evaluator_only"]) > 0.0
    )
    post_target_ok = (
        not args.post_target_avoidance
        or (avoidance_hold_seen and target_restore_seen)
    )
    return 0 if target_holding_reached and post_target_ok and ground_truth_clear and all(
        value is not None and value <= args.tolerance
        for value in final_errors.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
