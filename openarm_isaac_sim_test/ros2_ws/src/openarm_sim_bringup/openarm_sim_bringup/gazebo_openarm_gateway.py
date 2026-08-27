from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


def _project_root() -> Path:
    configured = os.environ.get("OPENARM_SIM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


class GazeboOpenArmGateway(Node):
    """Translate high-level GUI commands into standard trajectory actions."""

    def __init__(self) -> None:
        super().__init__("gazebo_openarm_gateway")
        self.declare_parameter("config", str(_project_root() / "config/openarm.yaml"))
        self.declare_parameter("home_duration_sec", 6.0)
        self._config = yaml.safe_load(
            Path(str(self.get_parameter("config").value)).read_text()
        )["robot"]
        self._duration = float(self.get_parameter("home_duration_sec").value)
        self._trajectory_clients = {
            side: ActionClient(
                self,
                FollowJointTrajectory,
                self._config["controller_actions"][side],
            )
            for side in ("left", "right")
        }
        self._goal_handles: dict[str, Any] = {}
        self._joint_names_seen: set[str] = set()
        self._task_state = "IDLE"
        self._safety_state = "SAFE"
        self._resume_home = False
        self._pending_results = 0
        self._task_pub = self.create_publisher(String, "/openarm/task/state", 10)
        self._safety_pub = self.create_publisher(String, "/openarm/safety/state", 10)
        self.create_subscription(String, "/openarm/task/command", self._on_task, 10)
        self.create_subscription(String, "/openarm/safety/command", self._on_safety, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_timer(0.25, self._publish_state)
        self.get_logger().info(
            "Gazebo OpenArm GUI gateway ready; no automatic motion is issued"
        )

    def _on_joints(self, message: JointState) -> None:
        self._joint_names_seen.update(message.name)

    def _on_task(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "home":
            self._start_home()
        elif command == "pause":
            self._pause()
        elif command == "resume":
            self._resume()
        elif command == "reset":
            self._cancel_all()
            self._resume_home = False
            self._task_state = "IDLE"
            self._safety_state = "SAFE"
        elif command.startswith("pick:"):
            self._task_state = "PICK_DISABLED_UNTIL_PHASE_3"

    def _on_safety(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "emergency_stop":
            self._cancel_all()
            self._resume_home = False
            self._task_state = "STOPPED"
            self._safety_state = "EMERGENCY_STOP"
        elif command == "pause":
            self._pause()
        elif command == "resume":
            self._resume()
        elif command == "reset":
            self._safety_state = "SAFE"

    def _controllers_ready(self) -> bool:
        return all(
            client.server_is_ready()
            for client in self._trajectory_clients.values()
        )

    def _start_home(self) -> None:
        required = {
            name
            for side in ("left", "right")
            for name in self._config["joint_names"][side]
        }
        if not required.issubset(self._joint_names_seen):
            self._task_state = "JOINT_STATES_NOT_READY"
            return
        if not self._controllers_ready():
            self._task_state = "CONTROLLER_NOT_READY"
            return
        if self._safety_state == "EMERGENCY_STOP":
            self._task_state = "RESET_REQUIRED"
            return
        self._cancel_all()
        self._pending_results = 2
        self._resume_home = True
        self._task_state = "HOME_RUNNING"
        self._safety_state = "SAFE"
        for side in ("left", "right"):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(self._config["joint_names"][side])
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in self._config["home"][side]]
            point.time_from_start = Duration(seconds=self._duration).to_msg()
            goal.trajectory.points = [point]
            future = self._trajectory_clients[side].send_goal_async(goal)
            future.add_done_callback(
                lambda response, selected=side: self._goal_response(selected, response)
            )

    def _goal_response(self, side: str, future: Any) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._task_state = f"{side.upper()}_HOME_REJECTED"
            self._resume_home = False
            return
        self._goal_handles[side] = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda response, selected=side: self._goal_result(selected, response)
        )

    def _goal_result(self, side: str, future: Any) -> None:
        response = future.result()
        self._goal_handles.pop(side, None)
        if response.status == GoalStatus.STATUS_CANCELED:
            return
        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._task_state = f"{side.upper()}_HOME_FAILED"
            self._resume_home = False
            return
        self._pending_results -= 1
        if self._pending_results <= 0:
            self._task_state = "HOME_REACHED"
            self._resume_home = False

    def _pause(self) -> None:
        was_running = self._task_state == "HOME_RUNNING"
        self._cancel_all()
        self._resume_home = was_running or self._resume_home
        self._task_state = "PAUSED"
        self._safety_state = "PAUSE"

    def _resume(self) -> None:
        if self._safety_state == "EMERGENCY_STOP":
            self._task_state = "RESET_REQUIRED"
            return
        should_resume = self._resume_home
        self._safety_state = "SAFE"
        if should_resume:
            self._start_home()
        else:
            self._task_state = "IDLE"

    def _cancel_all(self) -> None:
        for goal_handle in tuple(self._goal_handles.values()):
            goal_handle.cancel_goal_async()
        self._goal_handles.clear()

    def _publish_state(self) -> None:
        self._task_pub.publish(String(data=self._task_state))
        self._safety_pub.publish(String(data=self._safety_state))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboOpenArmGateway()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
