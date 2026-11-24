#!/usr/bin/env python3
"""Validate a real RGB-D moving-hand intrusion during OpenArm execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


class DynamicIntrusionValidator(Node):
    def __init__(self) -> None:
        super().__init__("validate_dynamic_intrusion")
        self.events: list[str] = []
        self.states: list[str] = []
        self.safety_states: list[str] = []
        self.hand_positions: list[list[float]] = []
        self.obstacle_centers: list[list[float]] = []
        self.minimum_distances: list[float] = []
        self.ground_truth_minimum_distances: list[float] = []
        self.left_target: np.ndarray | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.hand_target = self.create_publisher(
            PoseStamped, "/sim/hand/manual_target_pose", 10
        )
        self.hand_command = self.create_publisher(String, "/sim/hand/command", 10)
        self.task_command = self.create_publisher(String, "/openarm/task/command", 10)
        self.planner_mode = self.create_publisher(String, "/openarm/planner/mode", 10)
        self.obstacle_source = self.create_publisher(
            String, "/openarm/obstacle_source", 10
        )
        self.reset_estop = self.create_client(
            Trigger, "/openarm/safety/reset_estop"
        )
        self.create_subscription(String, "/openarm/events", self._event, 50)
        self.create_subscription(String, "/openarm/task/state", self._state, 10)
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(
            Float64, "/openarm/safety/min_distance", self._minimum_distance, 10
        )
        self.create_subscription(
            Float64,
            "/sim/ground_truth/min_distance",
            self._ground_truth_minimum_distance,
            10,
        )
        self.create_subscription(
            PoseStamped, "/sim/hand/actual_pose", self._hand_pose, 10
        )
        self.create_subscription(
            CollisionObject,
            "/openarm/dynamic_avoidance/collision_object",
            self._obstacle,
            10,
        )
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models, 10)

    def _event(self, message: String) -> None:
        self.events.append(message.data)

    def _state(self, message: String) -> None:
        if not self.states or message.data != self.states[-1]:
            self.states.append(message.data)

    def _safety(self, message: String) -> None:
        if not self.safety_states or message.data != self.safety_states[-1]:
            self.safety_states.append(message.data)

    def _minimum_distance(self, message: Float64) -> None:
        if np.isfinite(message.data):
            self.minimum_distances.append(float(message.data))

    def _ground_truth_minimum_distance(self, message: Float64) -> None:
        if np.isfinite(message.data):
            self.ground_truth_minimum_distances.append(float(message.data))

    def _hand_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        self.hand_positions.append([position.x, position.y, position.z])

    def _obstacle(self, message: CollisionObject) -> None:
        if (
            message.id == "perception_hand_obstacle"
            and message.operation != CollisionObject.REMOVE
            and message.primitive_poses
        ):
            position = message.primitive_poses[0].position
            self.obstacle_centers.append([position.x, position.y, position.z])

    def _models(self, message: ModelStates) -> None:
        try:
            index = message.name.index("left_target_cube")
        except ValueError:
            return
        position = message.pose[index].position
        self.left_target = np.asarray(
            [position.x, position.y, position.z], dtype=float
        )

    def publish_string(self, publisher: object, value: str) -> None:
        publisher.publish(String(data=value))

    def publish_hand_target(self, xyz: np.ndarray) -> None:
        message = PoseStamped()
        message.header.frame_id = "world"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(
            float, xyz
        )
        message.pose.orientation.w = 1.0
        self.hand_target.publish(message)

    def left_tcp(self) -> np.ndarray | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                "world",
                "openarm_left_hand_tcp",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            return None
        point = transform.transform.translation
        return np.asarray([point.x, point.y, point.z], dtype=float)


def spin_until(node: Node, timeout_sec: float, predicate: object) -> bool:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if callable(predicate) and bool(predicate()):
            return True
    return bool(predicate()) if callable(predicate) else False


def travel(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    values = np.asarray(points, dtype=float)
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--minimum-ground-truth-clearance", type=float, default=0.025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = DynamicIntrusionValidator()
    report: dict[str, object] = {}
    park = np.asarray([-0.10, -0.64, 0.32], dtype=float)
    original_hand: np.ndarray | None = None
    try:
        print("[STEP 1/5] Reading current hand pose", flush=True)
        spin_until(node, 2.0, lambda: bool(node.hand_positions))
        if node.hand_positions:
            original_hand = np.asarray(node.hand_positions[-1], dtype=float)
        node.publish_string(node.obstacle_source, "perception")
        node.publish_string(node.planner_mode, "dynamic")
        node.publish_string(node.hand_command, "auto_sweep:off")
        node.publish_string(node.hand_command, "speed:0.1")
        node.publish_hand_target(park)
        node.publish_string(node.hand_command, "manual:on")
        print("[STEP 2/5] Parking hand and homing OpenArm", flush=True)
        parked = spin_until(
            node,
            12.0,
            lambda: bool(
                node.hand_positions
                and np.linalg.norm(np.asarray(node.hand_positions[-1]) - park) < 0.03
            ),
        )
        if node.reset_estop.wait_for_service(timeout_sec=2.0):
            node.reset_estop.call_async(Trigger.Request())
        spin_until(
            node,
            5.0,
            lambda: bool(
                node.safety_states
                and node.safety_states[-1] not in {"EMERGENCY_STOP", "PAUSE"}
            ),
        )
        node.publish_string(node.task_command, "reset")
        node.publish_string(node.task_command, "home")
        homed = spin_until(
            node,
            25.0,
            lambda: bool(node.states and node.states[-1] == "HOME_REACHED"),
        )
        ready = spin_until(
            node,
            5.0,
            lambda: node.left_target is not None and node.left_tcp() is not None,
        )
        if not (parked and homed and ready):
            raise RuntimeError("could not establish parked-hand/home-arm baseline")

        tcp = node.left_tcp()
        assert tcp is not None and node.left_target is not None
        midpoint = 0.52 * tcp + 0.48 * node.left_target
        # This camera-calibrated band is already proven to yield a stable
        # hand mask and metric cloud.  The articulated-arm XY corridor check
        # deliberately reacts before the low hand reaches a robot link.
        midpoint[0] = -0.30
        midpoint[2] = 0.43
        sweep_start = midpoint.copy()
        sweep_end = midpoint.copy()
        # Begin outside the arm and camera mask.  Starting at -0.21 m made the
        # fast setup motion's predictive swept box touch the Home pose before
        # the commanded arm trajectory had even started.
        # Start close enough that the deliberately slow 0.02 m/s intrusion
        # reaches the active arm before its target/Home cycle completes.
        sweep_start[1] = -0.28
        sweep_end[1] = 0.30

        node.publish_string(node.hand_command, "speed:0.1")
        node.publish_hand_target(sweep_start)
        print("[STEP 3/5] Positioning hand at sweep start", flush=True)
        positioned = spin_until(
            node,
            15.0,
            lambda: bool(
                node.hand_positions
                and np.linalg.norm(
                    np.asarray(node.hand_positions[-1]) - sweep_start
                )
                < 0.03
            ),
        )
        # Let the measured-velocity predictor decay after the setup
        # move.  The actual intrusion below is independently commanded at
        # 0.02 m/s and must not inherit that setup velocity.
        spin_until(node, 2.0, lambda: False)
        if node.safety_states and node.safety_states[-1] == "EMERGENCY_STOP":
            if node.reset_estop.wait_for_service(timeout_sec=2.0):
                node.reset_estop.call_async(Trigger.Request())
            spin_until(
                node,
                8.0,
                lambda: bool(
                    node.safety_states
                    and node.safety_states[-1]
                    not in {"EMERGENCY_STOP", "PAUSE"}
                ),
            )

        node.events.clear()
        node.states.clear()
        node.safety_states.clear()
        node.hand_positions.clear()
        node.obstacle_centers.clear()
        node.minimum_distances.clear()
        node.ground_truth_minimum_distances.clear()
        node.publish_string(node.task_command, "move_left_target")
        print("[STEP 4/5] Left target/Home cycle started", flush=True)
        executing = spin_until(
            node,
            20.0,
            lambda: "EXECUTING" in node.states,
        )
        node.publish_string(node.hand_command, "speed:0.02")
        node.publish_hand_target(sweep_end)
        print("[STEP 5/5] Slow hand intrusion started at 0.02 m/s", flush=True)

        completed = False
        completion_count = 0
        completion_count_at_first_route: int | None = None
        deadline = time.monotonic() + max(float(args.timeout), 0.0)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            new_completion_count = sum(
                event.startswith("motion_complete,left_target,")
                for event in node.events
            )
            route_seen = any(
                event.startswith("dynamic_under_route_selected,")
                and ",side_forced_progress," in event
                for event in node.events
            )
            if route_seen and completion_count_at_first_route is None:
                completion_count_at_first_route = completion_count
            if new_completion_count > completion_count:
                completion_count = new_completion_count
                completed = True
                if (
                    route_seen
                    and completion_count_at_first_route is not None
                    and completion_count > completion_count_at_first_route
                ):
                    break
                if completion_count < 4:
                    # Keep an arm active while the deliberately slow hand
                    # crosses the route; each cycle still touches then Home.
                    node.publish_string(node.task_command, "move_left_target")
                    print(
                        f"[STEP 5/5] Repeating target/Home cycle {completion_count + 1}",
                        flush=True,
                    )
            if node.safety_states and node.safety_states[-1] == "EMERGENCY_STOP":
                break

        hand_motion = travel(node.hand_positions)
        obstacle_motion = travel(node.obstacle_centers)
        replan_events = [
            event
            for event in node.events
            if event.startswith(
                ("dynamic_spatial_replan,", "dynamic_under_route_selected,")
            )
        ]
        route_events = [
            event
            for event in node.events
            if event.startswith("dynamic_under_route_selected,")
        ]
        forced_route_events = [
            event for event in route_events if ",side_forced_progress," in event
        ]
        emergency = "EMERGENCY_STOP" in node.safety_states
        minimum_distance = (
            min(node.minimum_distances) if node.minimum_distances else None
        )
        ground_truth_minimum_distance = (
            min(node.ground_truth_minimum_distances)
            if node.ground_truth_minimum_distances
            else None
        )
        passed = bool(
            positioned
            and executing
            and hand_motion >= 0.25
            and obstacle_motion >= 0.06
            and replan_events
            and forced_route_events
            and completed
            and not emergency
            and minimum_distance is not None
            # Perception distance is intentionally conservative because the
            # tracked AABB is padded for planning.  Physical collision is
            # accepted/rejected only by evaluator-only ground truth below.
            and minimum_distance >= 0.0
            and ground_truth_minimum_distance is not None
            and ground_truth_minimum_distance
            > max(float(args.minimum_ground_truth_clearance), 0.0)
        )
        report = {
            "passed": passed,
            "planner": "MoveIt/OMPL spatial replan with live perception box",
            "obstacle_source": "perception_only",
            "hand_sweep_speed_mps": 0.02,
            "sweep_start_xyz": sweep_start.tolist(),
            "sweep_end_xyz": sweep_end.tolist(),
            "hand_motion_m": hand_motion,
            "perceived_obstacle_motion_m": obstacle_motion,
            "minimum_robot_obstacle_distance_m": minimum_distance,
            "ground_truth_minimum_distance_m_evaluator_only": (
                ground_truth_minimum_distance
            ),
            "required_ground_truth_clearance_m": max(
                float(args.minimum_ground_truth_clearance), 0.0
            ),
            "executing_before_intrusion": executing,
            "replan_events": replan_events,
            "route_events": route_events,
            "forced_progress_route_events": forced_route_events,
            "task_completed": completed,
            "completed_target_home_cycles": completion_count,
            "emergency_stop": emergency,
            "task_states": node.states,
            "safety_states": node.safety_states,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0 if passed else 1
    except RuntimeError as exc:
        report = {"passed": False, "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    finally:
        node.publish_string(node.hand_command, "speed:0.1")
        node.publish_hand_target(original_hand if original_hand is not None else park)
        node.publish_string(node.hand_command, "manual:on")
        spin_until(
            node,
            12.0,
            lambda: bool(
                node.hand_positions
                and np.linalg.norm(
                    np.asarray(node.hand_positions[-1])
                    - (original_hand if original_hand is not None else park)
                )
                < 0.04
            ),
        )
        if node.reset_estop.wait_for_service(timeout_sec=1.0):
            node.reset_estop.call_async(Trigger.Request())
        node.publish_string(node.task_command, "reset")
        node.publish_string(node.task_command, "home")
        spin_until(
            node,
            20.0,
            lambda: bool(node.states and node.states[-1] == "HOME_REACHED"),
        )
        node.publish_string(node.hand_command, "speed:0.1")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
