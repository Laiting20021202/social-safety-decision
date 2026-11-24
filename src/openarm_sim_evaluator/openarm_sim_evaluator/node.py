from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String

from openarm_sim.config import PROJECT_ROOT, load_yaml
from openarm_sim.metrics import EvaluationRecorder


class EvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__("openarm_sim_evaluator")
        self.declare_parameter("scenario", "no_obstacle")
        self.declare_parameter("mode", "ground_truth")
        self.declare_parameter("output_root", str(PROJECT_ROOT / "results"))
        scene_config = load_yaml("config/scene.yaml")
        self.evaluation = load_yaml("config/evaluation.yaml")
        self.recorder = EvaluationRecorder(
            Path(str(self.get_parameter("output_root").value)),
            str(self.get_parameter("scenario").value),
            int(scene_config["seed"]),
        )
        self.recorder.snapshot_configs(PROJECT_ROOT / "config")
        self.mode = str(self.get_parameter("mode").value)
        self.sim_time = 0.0
        self.wall_start = time.monotonic()
        self.clock_samples = 0
        self.min_distance = float("inf")
        self.in_collision = False
        self.sorted_cubes: set[str] = set()
        self.dropped_cubes: set[str] = set()
        self.latest_cube_states: list[dict[str, object]] = []
        self.done_seen_at: float | None = None
        self.last_velocity_scaling: float | None = None
        self.false_stop_clearance = float(
            load_yaml("config/safety_zones.yaml")["clearance"]["resume_distance_m"]
        )
        self.finalized = False
        self.create_subscription(Clock, "/clock", self._clock_callback, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state, 50)
        self.create_subscription(String, "/openarm/events", self._event, 50)
        self.create_subscription(String, "/openarm/task/state", self._task_state, 10)
        self.create_subscription(Float64, "/sim/ground_truth/min_distance", self._distance, 10)
        self.create_subscription(
            Float64, "/openarm/safety/velocity_scaling", self._velocity_scaling, 10
        )
        self.create_subscription(String, "/sim/ground_truth/cube_states", self._cube_states, 10)
        self.create_timer(0.5, self._check_timeout)

    def _clock_callback(self, message: Clock) -> None:
        self.sim_time = message.clock.sec + message.clock.nanosec * 1e-9
        self.clock_samples += 1

    def _joint_state(self, message: JointState) -> None:
        self.recorder.joint_sample(
            self.sim_time,
            list(message.name),
            list(message.position),
            list(message.velocity),
            list(message.effort),
        )
        if (
            "safety_command" in self.recorder.timings
            and message.velocity
            and max(abs(value) for value in message.velocity) < 0.01
        ):
            self.recorder.timings.setdefault("robot_stopped", self.sim_time)

    def _event(self, message: String) -> None:
        parts = message.data.split(",")
        event = parts[0]
        self.recorder.event(self.sim_time, event, payload=message.data)
        if event == "cube_placed" and len(parts) >= 2:
            # This is an attempt marker. Correct classification is independently
            # computed from simulator cube positions in _cube_states.
            self.recorder.event(self.sim_time, "place_attempt", cube=parts[1])
        elif event == "perception_output":
            self.recorder.timings.setdefault("perception_output", self.sim_time)
        elif event == "physical_grasp_success":
            self.recorder.counters["physical_grasp_successes"] += 1
        elif event == "magnetic_fallback":
            self.recorder.counters["magnetic_fallbacks"] += 1
        elif event == "safety_transition":
            self.recorder.timings.setdefault("safety_command", self.sim_time)
            if len(parts) >= 4 and parts[2] in {"PAUSE", "EMERGENCY_STOP"}:
                if float(parts[3]) > self.false_stop_clearance:
                    self.recorder.counters["false_stops"] += 1

    def _velocity_scaling(self, message: Float64) -> None:
        scaling = float(message.data)
        if self.last_velocity_scaling is None or scaling != self.last_velocity_scaling:
            self.recorder.event(self.sim_time, "velocity_scaling", value=scaling)
            self.last_velocity_scaling = scaling

    def _task_state(self, message: String) -> None:
        self.recorder.event(self.sim_time, "task_state", state=message.data)
        if message.data == "DONE":
            # Let released cubes settle and require a fresh simulator state
            # sample before writing final placement metrics.
            if self.done_seen_at is None:
                self.done_seen_at = self.sim_time

    def _distance(self, message: Float64) -> None:
        distance = float(message.data)
        self.min_distance = min(self.min_distance, distance)
        colliding = distance <= 1e-4
        if colliding and not self.in_collision:
            self.recorder.counters["collisions"] += 1
            self.recorder.event(self.sim_time, "collision")
        self.in_collision = colliding

    def _cube_states(self, message: String) -> None:
        states = json.loads(message.data)
        scene = load_yaml("config/scene.yaml")
        bins = scene["bins"]
        cube_size = float(scene["cubes"]["size"])
        count_per_color = int(scene["cubes"]["count_per_color"])
        dropped_height = float(self.evaluation["metrics"]["dropped_cube_height_m"])
        sorted_now: set[str] = set()
        dropped_now: set[str] = set()
        for cube in states:
            name = cube["name"]
            color = cube["color"]
            position = cube["position"]
            if position[2] < dropped_height:
                dropped_now.add(name)
            center = bins["centers"][color]
            inner = bins["inner_size"]
            xy_limit_x = max(0.0, float(inner[0]) / 2.0 - cube_size / 2.0 + 0.005)
            xy_limit_y = max(0.0, float(inner[1]) / 2.0 - cube_size / 2.0 + 0.005)
            inside_xy = (
                abs(position[0] - center[0]) <= xy_limit_x
                and abs(position[1] - center[1]) <= xy_limit_y
            )
            base_top = (
                float(center[2])
                - float(inner[2]) / 2.0
                + float(bins["base_thickness"]) / 2.0
            )
            minimum_z = base_top + cube_size / 2.0 - 0.015
            maximum_z = base_top + count_per_color * cube_size + 0.030
            if inside_xy and minimum_z <= position[2] <= maximum_z:
                sorted_now.add(name)
        self.latest_cube_states = states
        self.sorted_cubes = sorted_now
        self.dropped_cubes = dropped_now
        self.recorder.counters["correctly_sorted_cubes"] = len(self.sorted_cubes)
        self.recorder.counters["dropped_cubes"] = len(self.dropped_cubes)

    def _timeout(self) -> None:
        self.recorder.event(self.sim_time, "scenario_timeout")
        self._finalize(task_complete=False)

    def _check_timeout(self) -> None:
        if self.done_seen_at is not None and self.sim_time >= self.done_seen_at + 0.25:
            self._finalize(task_complete=True)
            return
        timeout = float(self.evaluation["scenario_timeout_sec"])
        if self.clock_samples and self.sim_time >= timeout:
            self._timeout()

    def _finalize(self, task_complete: bool) -> None:
        if self.finalized:
            return
        self.finalized = True
        wall_elapsed = max(time.monotonic() - self.wall_start, 1e-9)
        self.recorder.event(
            self.sim_time,
            "runtime_stats",
            simulator_fps=self.clock_samples / wall_elapsed,
            real_time_factor=self.sim_time / wall_elapsed,
            mode=self.mode,
        )
        minimum = None if self.min_distance == float("inf") else self.min_distance
        output = self.recorder.finalize(task_complete, minimum)
        (output / "cube_states_final.json").write_text(
            json.dumps(self.latest_cube_states, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(f"evaluation written to {output}")

    def destroy_node(self):
        self._finalize(task_complete=self.done_seen_at is not None)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
