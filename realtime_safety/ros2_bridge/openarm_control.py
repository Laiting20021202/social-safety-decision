from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger(__name__)
CUBES = (
    "red_cube_1",
    "red_cube_2",
    "green_cube_1",
    "green_cube_2",
    "blue_cube_1",
    "blue_cube_2",
)
PLANNER_MODES = ("moveit", "dynamic")
OBSTACLE_SOURCES = ("ground_truth", "perception")
GRASP_MODES = ("auto", "physical", "magnetic")
HAND_SPEEDS = (0.01, 0.02, 0.05, 0.1, 0.3, 0.6)


def validate_openarm_command(command: str, value: object | None) -> tuple[str, str]:
    name = str(command).strip().lower()
    payload = "" if value is None else str(value).strip().lower()
    if name == "openarm_pick":
        if payload not in CUBES:
            raise ValueError(f"Unknown cube: {value!r}")
        return "task", f"pick:{payload}"
    if name == "openarm_move_target":
        return "task", "move_target"
    if name in {
        "openarm_move_both_targets",
        "openarm_move_left_target",
        "openarm_move_right_target",
    }:
        return "task", name.removeprefix("openarm_")
    if name == "openarm_target":
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError("OpenArm target must contain world x/y/z")
        return "target_pose", ",".join(f"{item:.6f}" for item in values)
    if name in {"openarm_home", "openarm_pause", "openarm_resume", "openarm_reset"}:
        return "task", name.removeprefix("openarm_")
    if name == "openarm_estop":
        return "safety", "emergency_stop"
    if name == "openarm_planner":
        if payload not in PLANNER_MODES:
            raise ValueError(f"Unknown planner mode: {value!r}")
        return "planner", payload
    if name == "openarm_obstacle_source":
        if payload not in OBSTACLE_SOURCES:
            raise ValueError(f"Unknown obstacle source: {value!r}")
        return "obstacle", payload
    if name == "openarm_grasp":
        if payload not in GRASP_MODES:
            raise ValueError(f"Unknown grasp mode: {value!r}")
        return "grasp", payload
    if name == "openarm_hand_enable":
        return "hand", "manual:on" if bool(value) else "manual:off"
    if name == "openarm_hand_auto_sweep":
        return "hand", "auto_sweep:on" if bool(value) else "auto_sweep:off"
    if name == "openarm_hand_speed":
        speed = float(value)
        if speed not in HAND_SPEEDS:
            raise ValueError(f"Unsupported manual hand speed: {value!r}")
        return "hand", f"speed:{speed:.2f}"
    if name == "openarm_hand_withdraw":
        return "hand", "withdraw"
    if name == "openarm_hand_reset":
        return "hand", "reset_hand"
    if name == "openarm_hand_preview":
        return "hand", "perception_preview"
    if name == "openarm_hand_intrusion":
        return "hand", "sudden_intrusion"
    if name == "openarm_hand_target":
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError("Manual hand target must contain x/y/z")
        return "hand_pose", ",".join(f"{item:.6f}" for item in values)
    raise ValueError(f"Unknown OpenArm GUI command: {command!r}")


class OpenArmControlBridge:
    """Publish validated high-level GUI requests; never publishes joint targets."""

    def __init__(self, on_status: Callable[[str, str], None] | None = None) -> None:
        self.on_status = on_status
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._publishers: dict[str, Any] = {}
        self._subscriptions: list[Any] = []

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from std_msgs.msg import String
        from geometry_msgs.msg import PoseStamped

        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node("realtime_safety_openarm_control", context=runtime.context)
        publishers = {
            "task": node.create_publisher(String, "/openarm/task/command", 10),
            "safety": node.create_publisher(String, "/openarm/safety/command", 10),
            "planner": node.create_publisher(String, "/openarm/planner/mode", 10),
            "obstacle": node.create_publisher(String, "/openarm/obstacle_source", 10),
            "grasp": node.create_publisher(String, "/openarm/grasp/mode", 10),
            "hand": node.create_publisher(String, "/sim/hand/command", 10),
            "hand_pose": node.create_publisher(
                PoseStamped, "/sim/hand/manual_target_pose", 10
            ),
            "target_pose": node.create_publisher(
                PoseStamped, "/openarm/target_pose", 10
            ),
        }
        subscriptions = [
            node.create_subscription(
                String,
                "/openarm/task/state",
                lambda message: self._status("task", message.data),
                10,
            ),
            node.create_subscription(
                String,
                "/openarm/safety/state",
                lambda message: self._status("safety", message.data),
                10,
            ),
            node.create_subscription(
                String,
                "/openarm/dynamic_avoidance/status",
                lambda message: self._status("dynamic", message.data),
                10,
            ),
            node.create_subscription(
                String,
                "/openarm/grasp/status",
                lambda message: self._status("grasp", message.data),
                10,
            ),
            node.create_subscription(
                String,
                "/sim/hand/status",
                lambda message: self._status("hand", message.data),
                10,
            ),
        ]
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._publishers = publishers
        self._subscriptions = subscriptions
        LOGGER.info("OpenArm high-level GUI control bridge started")

    def send(self, command: str, value: object | None = None) -> None:
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String

        channel, payload = validate_openarm_command(command, value)
        publisher = self._publishers.get(channel)
        if publisher is None:
            raise RuntimeError("OpenArm control bridge is not started")
        if channel in {"hand_pose", "target_pose"}:
            x, y, z = (float(item) for item in payload.split(","))
            message = PoseStamped()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = "world"
            message.pose.position.x = x
            message.pose.position.y = y
            message.pose.position.z = z
            if channel == "target_pose":
                # Match the validated downward-facing OpenArm tool
                # orientation. The GUI controls position only.
                message.pose.orientation.x = 1.0
            else:
                message.pose.orientation.w = 1.0
            publisher.publish(message)
        elif command in {"openarm_hand_preview", "openarm_hand_intrusion"}:
            scenario = (
                "perception_preview"
                if command == "openarm_hand_preview"
                else "sudden_intrusion"
            )
            publisher.publish(String(data=f"scenario:{scenario}"))
            publisher.publish(String(data="trigger"))
        else:
            publisher.publish(String(data=payload))
        LOGGER.info(
            "OpenArm GUI command: %s=%r -> %s:%s (matched subscribers=%d)",
            command,
            value,
            channel,
            payload,
            publisher.get_subscription_count(),
        )
        # Reset and resume affect both the task state machine and the latched
        # safety supervisor. They remain high-level commands, never joints.
        if command == "openarm_reset":
            self._publishers["safety"].publish(String(data="reset"))
            self._publishers["hand"].publish(String(data="reset"))
        elif command == "openarm_pause":
            self._publishers["safety"].publish(String(data="pause"))
        elif command == "openarm_resume":
            self._publishers["safety"].publish(String(data="resume"))
        self._status("command", payload)

    def _status(self, key: str, value: str) -> None:
        if self.on_status is not None:
            self.on_status(str(key), str(value))

    def close(self) -> None:
        if self._node is None:
            return
        from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

        runtime, node = self._runtime, self._node
        self._runtime = None
        self._node = None
        self._publishers = {}
        self._subscriptions = []
        if runtime is not None:
            runtime.remove_node(node)
        node.destroy_node()
        if runtime is not None:
            release_ros2_runtime(runtime)


__all__ = [
    "CUBES",
    "HAND_SPEEDS",
    "GRASP_MODES",
    "OBSTACLE_SOURCES",
    "PLANNER_MODES",
    "OpenArmControlBridge",
    "validate_openarm_command",
]
