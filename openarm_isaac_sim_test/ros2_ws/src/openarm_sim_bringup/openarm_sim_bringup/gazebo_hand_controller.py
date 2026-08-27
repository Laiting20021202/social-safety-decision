from __future__ import annotations

import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import yaml
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


def _project_root() -> Path:
    configured = os.environ.get("OPENARM_SIM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def bounded_sim_dt(previous_sec: float, current_sec: float) -> float:
    """Return a safe simulation-time step without jumps after clock resets."""

    previous = float(previous_sec)
    current = float(current_sec)
    if not np.isfinite((previous, current)).all() or current <= previous:
        return 0.0
    return min(current - previous, 0.1)


class GazeboHandController(Node):
    """Rate-limited world-frame target controller for the Gazebo hand model."""

    def __init__(self) -> None:
        super().__init__("gazebo_hand_controller")
        self.declare_parameter(
            "config", str(_project_root() / "config/hand_scenarios.yaml")
        )
        self.declare_parameter("model_name", "human_hand")
        self.declare_parameter("update_rate_hz", 30.0)
        document = yaml.safe_load(
            Path(str(self.get_parameter("config").value)).read_text()
        )
        self._defaults = document["defaults"]
        self._scenarios = document["scenarios"]
        self._speed_profiles = {
            float(value) for value in document.get("speed_profiles_mps", ())
        }
        if not self._speed_profiles:
            self._speed_profiles = {float(self._defaults["linear_speed"])}
        self._model_name = str(self.get_parameter("model_name").value)
        self._workspace_min = np.asarray(
            self._defaults["manual_workspace_min"], dtype=float
        )
        self._workspace_max = np.asarray(
            self._defaults["manual_workspace_max"], dtype=float
        )
        self._parked = np.asarray(
            self._scenarios["no_obstacle"]["path_waypoints"][0], dtype=float
        )
        self._current = self._parked.copy()
        self._actual = self._parked.copy()
        self._actual_orientation = np.asarray([0.0, 0.0, 0.0, 1.0])
        self._orientation = self._actual_orientation.copy()
        self._target = self._parked.copy()
        self._manual_target = self._parked.copy()
        self._waypoints: deque[np.ndarray] = deque()
        self._scenario_name = "no_obstacle"
        self._speed = float(self._defaults["linear_speed"])
        self._manual_enabled = False
        axis_name = str(self._defaults.get("auto_sweep_axis", "y")).lower()
        self._auto_sweep_axis = {"x": 0, "y": 1, "z": 2}.get(axis_name, 1)
        limits = np.asarray(
            self._defaults.get("auto_sweep_limits_m", [-0.50, 0.35]),
            dtype=float,
        )
        if limits.shape != (2,) or not np.isfinite(limits).all():
            raise ValueError("auto_sweep_limits_m must contain two finite values")
        self._auto_sweep_limits = np.sort(limits)
        self._auto_sweep_enabled = False
        self._paused = False
        self._pending = None
        # Hand speed is specified in simulated metres/second.  Gazebo can run
        # far below real time while RGB-D inference is active; wall-clock
        # integration would outrun the sensor and safety loops.
        self._last_sim_time = self.get_clock().now().nanoseconds * 1e-9
        self._last_pose_publish_wall = 0.0
        self._failure_count = 0

        self._client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self._actual_pub = self.create_publisher(
            PoseStamped, "/sim/hand/actual_pose", 10
        )
        self._ground_truth_pub = self.create_publisher(
            PoseStamped, "/sim/ground_truth/hand_pose", 10
        )
        self._status_pub = self.create_publisher(String, "/sim/hand/status", 10)
        self.create_subscription(
            PoseStamped,
            "/sim/hand/manual_target_pose",
            self._on_manual_target,
            10,
        )
        self.create_subscription(
            String, "/sim/hand/command", self._on_command, 10
        )
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_model_states, 10
        )
        update_rate = max(float(self.get_parameter("update_rate_hz").value), 1.0)
        self.create_timer(1.0 / update_rate, self._tick)
        self.get_logger().info(
            "Gazebo hand controller ready: target_pose -> smooth model motion"
        )

    def _on_model_states(self, message: ModelStates) -> None:
        try:
            index = message.name.index(self._model_name)
        except ValueError:
            return
        position = message.pose[index].position
        orientation = message.pose[index].orientation
        actual = np.asarray([position.x, position.y, position.z], dtype=float)
        actual_orientation = np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w], dtype=float
        )
        if not np.isfinite(actual).all():
            return
        orientation_norm = float(np.linalg.norm(actual_orientation))
        if not np.isfinite(orientation_norm) or orientation_norm <= 1e-9:
            actual_orientation = np.asarray([0.0, 0.0, 0.0, 1.0])
        else:
            actual_orientation /= orientation_norm
        self._actual = actual
        self._actual_orientation = actual_orientation
        # A direct Gazebo transform edit is authoritative while the controller
        # is idle.  Synchronizing here prevents a later GUI command from
        # snapping the hand back to an obsolete internal pose.
        idle = (
            self._pending is None
            and not self._waypoints
            and float(np.linalg.norm(self._target - self._current)) <= 1e-5
        )
        if idle:
            self._current = actual.copy()
            self._target = actual.copy()
            self._orientation = actual_orientation.copy()
            if not self._manual_enabled:
                self._manual_target = actual.copy()
        if time.monotonic() - self._last_pose_publish_wall >= 0.1:
            self._publish_pose(actual, actual_orientation)

    def _on_manual_target(self, message: PoseStamped) -> None:
        if message.header.frame_id not in {"", "world"}:
            self.get_logger().warning(
                f"Ignoring hand target in frame {message.header.frame_id!r}"
            )
            return
        requested = np.array(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ],
            dtype=float,
        )
        if not np.isfinite(requested).all():
            return
        requested_orientation = np.asarray(
            [
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ],
            dtype=float,
        )
        orientation_norm = float(np.linalg.norm(requested_orientation))
        if np.isfinite(orientation_norm) and orientation_norm > 1e-9:
            self._orientation = requested_orientation / orientation_norm
        self._manual_target = np.clip(
            requested, self._workspace_min, self._workspace_max
        )
        if self._manual_enabled:
            self._waypoints.clear()
            self._target = self._manual_target.copy()

    def _on_command(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "manual:on":
            self._auto_sweep_enabled = False
            self._manual_enabled = True
            self._paused = False
            self._waypoints.clear()
            self._target = self._manual_target.copy()
            self._publish_status("MANUAL")
        elif command == "manual:off":
            self._manual_enabled = False
            self._waypoints.clear()
            self._target = self._current.copy()
            self._publish_status("HOLD")
        elif command == "auto_sweep:on":
            self._start_auto_sweep()
        elif command == "auto_sweep:off":
            self._auto_sweep_enabled = False
            self._waypoints.clear()
            self._target = self._current.copy()
            self._publish_status("AUTO SWEEP OFF")
        elif command.startswith("speed:"):
            value = float(command.split(":", 1)[1])
            if value in self._speed_profiles:
                self._speed = value
                self._publish_status(f"SPEED {value:.2f}")
            else:
                profiles = ", ".join(f"{item:.2f}" for item in sorted(self._speed_profiles))
                self.get_logger().warning(
                    "Rejected unsupported hand speed %.3f; allowed: %s",
                    value,
                    profiles,
                )
                self._publish_status(f"SPEED REJECTED {value:.3f}")
        elif command.startswith("scenario:"):
            name = command.split(":", 1)[1]
            if name in self._scenarios:
                self._scenario_name = name
                self._publish_status(f"SCENARIO {name}")
        elif command in {"trigger", "trigger_hand", "start"}:
            self._auto_sweep_enabled = False
            self._start_scenario(self._scenario_name)
        elif command in {"withdraw", "reset", "reset_hand"}:
            self._auto_sweep_enabled = False
            self._waypoints.clear()
            self._target = self._parked.copy()
            self._paused = False
            self._publish_status("WITHDRAW" if command == "withdraw" else "RESET")
        elif command == "pause":
            self._paused = True
            self._publish_status("PAUSED")
        elif command == "resume":
            self._paused = False
            self._publish_status("MANUAL" if self._manual_enabled else "RUNNING")

    def _start_auto_sweep(self) -> None:
        """Continuously sweep left/right while preserving current X/Z."""

        self._manual_enabled = False
        self._auto_sweep_enabled = True
        self._paused = False
        self._waypoints.clear()
        self._current = self._actual.copy()
        self._orientation = self._actual_orientation.copy()
        low, high = self._auto_sweep_limits
        axis_value = float(self._current[self._auto_sweep_axis])
        next_value = high if abs(axis_value - low) <= abs(axis_value - high) else low
        self._target = self._current.copy()
        self._target[self._auto_sweep_axis] = next_value
        self._target = np.clip(
            self._target, self._workspace_min, self._workspace_max
        )
        self._publish_status(
            f"AUTO SWEEP ON FIXED_Z {self._current[2]:.3f}"
        )

    def _start_scenario(self, name: str) -> None:
        scenario = self._scenarios.get(name)
        if not scenario:
            return
        speed = scenario.get("linear_speed", self._defaults["linear_speed"])
        self._speed = float(speed)
        self._waypoints = deque(
            np.clip(np.asarray(point, dtype=float), self._workspace_min, self._workspace_max)
            for point in scenario.get("path_waypoints", ())
        )
        if self._waypoints:
            self._target = self._waypoints.popleft()
            self._paused = False
            self._publish_status(f"RUNNING {name}")

    def _tick(self) -> None:
        wall_now = time.monotonic()
        sim_now = self.get_clock().now().nanoseconds * 1e-9
        dt = bounded_sim_dt(self._last_sim_time, sim_now)
        self._last_sim_time = sim_now
        if self._pending is not None:
            if not self._pending.done():
                return
            try:
                response = self._pending.result()
            except Exception as exc:  # ROS service transport failure is actionable.
                self._failure_count += 1
                if self._failure_count <= 3:
                    self.get_logger().error(f"hand state service failed: {exc}")
                self._pending = None
                return
            self._pending = None
            if response is None or not response.success:
                self._failure_count += 1
                if self._failure_count <= 3:
                    self.get_logger().error("Gazebo rejected human_hand state")
                return
            self._failure_count = 0
            self._publish_pose(self._actual)
        if self._paused or not self._client.service_is_ready():
            return
        delta = self._target - self._current
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-5:
            if self._waypoints:
                self._target = self._waypoints.popleft()
            elif self._auto_sweep_enabled:
                low, high = self._auto_sweep_limits
                axis = self._auto_sweep_axis
                self._target = self._current.copy()
                self._target[axis] = (
                    low if abs(float(self._current[axis]) - high) <= 1e-4 else high
                )
            elif wall_now - self._last_pose_publish_wall >= 0.25:
                self._publish_pose(self._actual)
            return
        step = min(distance, self._speed * max(dt, 1e-3))
        self._current += delta * (step / distance)
        request = SetEntityState.Request()
        request.state.name = self._model_name
        request.state.reference_frame = "world"
        request.state.pose.position.x = float(self._current[0])
        request.state.pose.position.y = float(self._current[1])
        request.state.pose.position.z = float(self._current[2])
        request.state.pose.orientation.x = float(self._orientation[0])
        request.state.pose.orientation.y = float(self._orientation[1])
        request.state.pose.orientation.z = float(self._orientation[2])
        request.state.pose.orientation.w = float(self._orientation[3])
        # SetEntityState already receives a rate-limited pose every tick.
        # Supplying the same velocity as a twist made Gazebo continue
        # integrating after the final waypoint, so the hand drifted metres
        # away from both RGB-D and its collision cloud.  Leave twist exactly
        # zero; motion remains smooth because pose increments are speed-bound.
        self._pending = self._client.call_async(request)
    def _publish_pose(
        self,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        source = self._actual if position is None else position
        source_orientation = (
            self._actual_orientation if orientation is None else orientation
        )
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = float(source[0])
        message.pose.position.y = float(source[1])
        message.pose.position.z = float(source[2])
        message.pose.orientation.x = float(source_orientation[0])
        message.pose.orientation.y = float(source_orientation[1])
        message.pose.orientation.z = float(source_orientation[2])
        message.pose.orientation.w = float(source_orientation[3])
        self._actual_pub.publish(message)
        self._ground_truth_pub.publish(message)
        self._last_pose_publish_wall = time.monotonic()

    def _publish_status(self, value: str) -> None:
        self._status_pub.publish(String(data=value))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboHandController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
