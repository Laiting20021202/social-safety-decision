from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

from openarm_sim.config import PROJECT_ROOT, load_yaml
from openarm_sim.scene_model import CubeSpec, deterministic_cube_layout
from openarm_sim.state_machine import SafetyState, TaskState, TaskStateMachine

from .grasp import grasp_decision


class SortingTaskNode(Node):
    def __init__(self) -> None:
        super().__init__("openarm_sorting_task")
        self.declare_parameter("config", str(PROJECT_ROOT / "config/sorting_task.yaml"))
        self.declare_parameter("auto_start", True)
        self.config = _load_path(self.get_parameter("config").value)
        self.robot_config = load_yaml("config/openarm.yaml")["robot"]
        self.scene_config = load_yaml("config/scene.yaml")
        self.machine = TaskStateMachine()
        self.cubes = deterministic_cube_layout(self.scene_config)
        self.current_cube_index = 0
        self.selected_only = False
        self._home_only = False
        self.planner_mode = "moveit"
        self.grasp_mode = str(self.config["grasp"]["mode"]).lower()
        self.grasp_status = self.grasp_mode.upper()
        self._physical_grasp_attempt = 0
        self._current_grasp_assist = ""
        self.busy = False
        self.started = bool(self.get_parameter("auto_start").value)
        self._joint_state_ready = False
        self._planning_failures = 0
        self._retry_not_before_ns = 0
        self._fatal_error = False
        self._move_goal_handle: Any | None = None
        self._trajectory_goal_handle: Any | None = None
        self._state_pub = self.create_publisher(String, "/openarm/task/state", 10)
        self._event_pub = self.create_publisher(String, "/openarm/events", 50)
        self._grasp_status_pub = self.create_publisher(
            String, "/openarm/grasp/status", 10
        )
        self._dynamic_input_pub = self.create_publisher(
            __import__("trajectory_msgs.msg", fromlist=["JointTrajectory"]).JointTrajectory,
            "/openarm/dynamic_avoidance/input_trajectory",
            10,
        )
        self._dynamic_target_pub = self.create_publisher(
            PoseStamped, "/openarm/dynamic_avoidance/target_pose", 10
        )
        self._planning_scene_pub = self.create_publisher(
            __import__("moveit_msgs.msg", fromlist=["PlanningScene"]).PlanningScene,
            "/planning_scene",
            10,
        )
        self.create_subscription(String, "/openarm/safety/state", self._on_safety, 10)
        self.create_subscription(String, "/openarm/task/command", self._on_command, 10)
        self.create_subscription(String, "/openarm/planner/mode", self._on_planner_mode, 10)
        self.create_subscription(String, "/openarm/grasp/mode", self._on_grasp_mode, 10)
        self.create_subscription(
            __import__("trajectory_msgs.msg", fromlist=["JointTrajectory"]).JointTrajectory,
            "/openarm/dynamic_avoidance/trajectory",
            self._dynamic_trajectory,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, self.config["move_group_action"])
        trajectory_action = self.robot_config["controller_actions"]["left"]
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, trajectory_action
        )
        action = self.robot_config["controller_actions"]["left_gripper"]
        self.gripper_client = ActionClient(self, GripperCommand, action)
        self.create_timer(0.05, self._tick)
        self._publish_world_cubes()

    def _tick(self) -> None:
        self._state_pub.publish(String(data=self.machine.task_state.value))
        self._grasp_status_pub.publish(String(data=self.grasp_status))
        if not self.started or self.busy or not self._joint_state_ready or self._fatal_error:
            return
        if self.get_clock().now().nanoseconds < self._retry_not_before_ns:
            return
        if self.machine.safety_state not in {SafetyState.SAFE, SafetyState.WARNING}:
            return
        state = self.machine.task_state
        if state is TaskState.DONE:
            return
        cube = self._current_cube()
        if state is TaskState.SELECT_OBJECT:
            if cube is None:
                self.machine.task_state = TaskState.DONE
                self._emit("task_complete")
            else:
                self._emit("object_selected", cube.name, cube.color)
                self.machine.advance()
            return
        if state is TaskState.NEXT_OBJECT:
            if self.selected_only:
                self.machine.task_state = TaskState.DONE
                self.started = False
                self._emit("selected_task_complete", self._current_cube().name)
                return
            self.current_cube_index += 1
            if self.current_cube_index >= len(self.cubes):
                self.machine.task_state = TaskState.DONE
                self._emit("task_complete")
            else:
                self.machine.reset_for_next_object()
            return
        if state is TaskState.PLACE:
            self._send_pose(self._pose_for_state(state, cube))
            return
        self._send_motion_for_state(state, cube)

    def _send_motion_for_state(self, state: TaskState, cube: CubeSpec | None) -> None:
        if state is TaskState.HOME:
            self._send_joint_goal(self.robot_config["home"]["left"])
        elif state in {
            TaskState.PRE_GRASP,
            TaskState.GRASP,
            TaskState.LIFT,
            TaskState.TRANSIT,
            TaskState.RETREAT,
        }:
            assert cube is not None
            if state is TaskState.GRASP:
                # The selected object is intentionally contacted during the
                # grasp descent.  Remove only that cube from the world model;
                # every other cube, bin, table and dynamic obstacle remains
                # collision checked.  It is reintroduced as an attached body
                # after the gripper action succeeds.
                self._remove_world_cube(cube)
            self._send_pose(self._pose_for_state(state, cube))
        else:
            raise RuntimeError(f"unhandled task state: {state.value}")

    def _send_joint_goal(self, positions: list[float]) -> None:
        names = self.robot_config["joint_names"]["left"]
        constraints = Constraints()
        for name, position in zip(names, positions, strict=True):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        self._send_moveit_goal(constraints)

    def _send_pose(self, pose: PoseStamped) -> None:
        self._dynamic_target_pub.publish(pose)
        tolerance = float(self.config["pose_tolerance_m"])
        position = PositionConstraint()
        position.header = pose.header
        position.link_name = self.config["end_effector_link"]
        position.weight = 1.0
        region = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[tolerance * 2.0] * 3)
        position.constraint_region.primitives.append(region)
        position.constraint_region.primitive_poses.append(pose.pose)
        orientation = OrientationConstraint()
        orientation.header = pose.header
        orientation.link_name = self.config["end_effector_link"]
        orientation.orientation = pose.pose.orientation
        orientation.absolute_x_axis_tolerance = float(self.config["orientation_tolerance_rad"])
        orientation.absolute_y_axis_tolerance = float(self.config["orientation_tolerance_rad"])
        orientation.absolute_z_axis_tolerance = float(self.config["orientation_tolerance_rad"])
        orientation.weight = 1.0
        constraints = Constraints(position_constraints=[position], orientation_constraints=[orientation])
        self._send_moveit_goal(constraints)

    def _send_moveit_goal(self, constraints: Constraints) -> None:
        if not self.move_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().error("/move_action is unavailable; install/start MoveIt 2")
            return
        goal = MoveGroup.Goal()
        goal.request.group_name = self.config["active_group"]
        goal.request.num_planning_attempts = int(self.config["planning_attempts"])
        goal.request.allowed_planning_time = float(self.config["planning_time_sec"])
        goal.request.max_velocity_scaling_factor = float(self.config["velocity_scaling"])
        goal.request.max_acceleration_scaling_factor = float(self.config["acceleration_scaling"])
        goal.request.pipeline_id = self.config["planning_pipeline"]
        goal.request.planner_id = self.config["planner_id"]
        goal.request.goal_constraints = [constraints]
        # An empty absolute start_state is invalid. Mark it as a diff so
        # MoveIt uses the latest state from its CurrentStateMonitor.
        goal.request.start_state.is_diff = True
        # MoveIt plans; the standard FollowJointTrajectory server in Isaac
        # executes. This avoids coupling to a particular MoveIt controller
        # manager plugin while preserving the canonical ROS control contract.
        goal.planning_options = PlanningOptions(plan_only=True, look_around=False, replan=False)
        self.busy = True
        future = self.move_client.send_goal_async(goal)
        future.add_done_callback(self._move_goal_response)

    def _move_goal_response(self, future: Any) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.busy = False
            self._emit("planning_failed", self.machine.task_state.value)
            self._record_planning_failure()
            return
        self._move_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._move_result)

    def _move_result(self, future: Any) -> None:
        result = future.result().result
        self._move_goal_handle = None
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.busy = False
            self._emit("planning_or_control_failed", str(result.error_code.val))
            self._record_planning_failure()
            return
        if self.machine.safety_state in {SafetyState.PAUSE, SafetyState.EMERGENCY_STOP}:
            self.busy = False
            return
        self._send_trajectory(result.planned_trajectory.joint_trajectory)

    def _send_trajectory(self, trajectory: Any) -> None:
        if self.planner_mode == "dynamic":
            self._dynamic_input_pub.publish(trajectory)
            self._emit("dynamic_trajectory_requested", self.machine.task_state.value)
            return
        self._execute_trajectory(trajectory)

    def _dynamic_trajectory(self, trajectory: Any) -> None:
        if self.planner_mode != "dynamic" or not self.busy:
            return
        self._emit("dynamic_trajectory_received", self.machine.task_state.value)
        self._execute_trajectory(trajectory)

    def _execute_trajectory(self, trajectory: Any) -> None:
        if not trajectory.points:
            self.busy = False
            self._emit("empty_trajectory", self.machine.task_state.value)
            self.machine.set_safety(SafetyState.REPLAN)
            return
        if not self.trajectory_client.wait_for_server(timeout_sec=0.2):
            self.busy = False
            self.get_logger().error("left FollowJointTrajectory action is unavailable")
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        duration = trajectory.points[-1].time_from_start
        self.get_logger().info(
            f"sending {self.machine.task_state.value} trajectory: "
            f"{len(trajectory.points)} points, "
            f"duration={duration.sec + duration.nanosec * 1e-9:.3f}s"
        )
        self.trajectory_client.send_goal_async(goal).add_done_callback(
            self._trajectory_goal_response
        )

    def _trajectory_goal_response(self, future: Any) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.busy = False
            self._emit("trajectory_rejected", self.machine.task_state.value)
            self.machine.set_safety(SafetyState.REPLAN)
            return
        self._trajectory_goal_handle = goal_handle
        self.get_logger().info(
            f"trajectory accepted for state {self.machine.task_state.value}"
        )
        goal_handle.get_result_async().add_done_callback(self._trajectory_result)

    def _trajectory_result(self, future: Any) -> None:
        response = future.result()
        self._trajectory_goal_handle = None
        self.busy = False
        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            if self.machine.safety_state in {
                SafetyState.PAUSE,
                SafetyState.EMERGENCY_STOP,
            }:
                return
            self._emit("trajectory_failed", str(response.result.error_code))
            self._record_planning_failure()
            return
        self._planning_failures = 0
        self._motion_complete()

    def _record_planning_failure(self) -> None:
        self._planning_failures += 1
        if self._planning_failures >= int(self.config["maximum_replan_attempts"]):
            self._emit("maximum_replans_exceeded", self.machine.task_state.value)
            self._fatal_error = True
            self.machine.set_safety(SafetyState.REPLAN)
            return
        self.machine.set_safety(SafetyState.REPLAN)
        backoff_ns = int(float(self.config["replan_backoff_sec"]) * 1e9)
        self._retry_not_before_ns = self.get_clock().now().nanoseconds + backoff_ns

    def _motion_complete(self) -> None:
        state = self.machine.task_state
        if state is TaskState.HOME and self._home_only:
            self._home_only = False
            self.started = False
            self._emit("home_reached")
        elif state is TaskState.GRASP:
            self._physical_grasp_attempt = 0
            self._start_grasp_attempt()
        elif state is TaskState.PLACE:
            self._command_gripper(float(self.robot_config["gripper"]["open_position"]), closed=False)
        else:
            if state is TaskState.RETREAT:
                cube = self._current_cube()
                if cube is not None:
                    # The released cube initially remains between the open
                    # fingers. Add it to the world collision model only after
                    # the hand has cleared; PhysX continues simulating it
                    # throughout the release and retreat.
                    self._add_placed_cube(cube)
            self.machine.advance()

    def _start_grasp_attempt(self) -> None:
        self._physical_grasp_attempt += 1
        maximum = int(self.config["grasp"]["physical_attempts"])
        if self.grasp_mode == "magnetic":
            self.grasp_status = "MAGNETIC"
        else:
            self.grasp_status = (
                f"PHYSICAL ATTEMPT {self._physical_grasp_attempt}/{maximum}"
            )
        self._command_gripper(
            float(self.robot_config["gripper"]["closed_position"]), closed=True
        )

    def _command_gripper(
        self, position: float, closed: bool, retry_physical: bool = False
    ) -> None:
        if not self.gripper_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().error("left gripper action is unavailable")
            return
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = float(self.robot_config["gripper"]["max_effort"])
        self.busy = True
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(
            lambda response: self._gripper_goal_response(
                response, closed, retry_physical
            )
        )

    def _gripper_goal_response(
        self, future: Any, closed: bool, retry_physical: bool
    ) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.busy = False
            self.machine.set_safety(SafetyState.EMERGENCY_STOP)
            return
        goal_handle.get_result_async().add_done_callback(
            lambda response: self._gripper_result(
                response, closed, retry_physical
            )
        )

    def _gripper_result(
        self, future: Any, closed: bool, retry_physical: bool
    ) -> None:
        response = future.result()
        self.busy = False
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self._emit("gripper_failed", str(response.status))
            self.machine.set_safety(SafetyState.EMERGENCY_STOP)
            return
        cube = self._current_cube()
        if cube is None:
            return
        if retry_physical and not closed:
            self._start_grasp_attempt()
            return
        if closed:
            result = response.result
            maximum = int(self.config["grasp"]["physical_attempts"])
            decision = grasp_decision(
                self.grasp_mode,
                self._physical_grasp_attempt,
                maximum,
                stalled=bool(result.stalled),
                reached_goal=bool(result.reached_goal),
            )
            if decision == "physical":
                self._complete_grasp(cube, assist="physical")
            elif decision == "magnetic":
                self._complete_grasp(cube, assist="magnetic")
            elif decision == "retry":
                self._emit(
                    "physical_grasp_attempt_failed",
                    cube.name,
                    str(self._physical_grasp_attempt),
                )
                self._command_gripper(
                    float(self.robot_config["gripper"]["open_position"]),
                    closed=False,
                    retry_physical=True,
                )
            else:
                self.grasp_status = "PHYSICAL FAILED"
                self._emit("physical_grasp_failed", cube.name)
                self._fatal_error = True
                self.machine.set_safety(SafetyState.REPLAN)
        else:
            self._detach_cube(cube)
            if self._current_grasp_assist == "magnetic":
                self._emit("magnetic_detach", cube.name)
            self._emit("cube_detached", cube.name, cube.color)
            self._emit("cube_placed", cube.name, cube.color)
            self._current_grasp_assist = ""
            self.machine.advance()

    def _complete_grasp(self, cube: CubeSpec, assist: str) -> None:
        self._current_grasp_assist = assist
        self._attach_cube(cube)
        self._emit("cube_attached", cube.name, cube.color)
        if assist == "physical":
            self.grasp_status = "PHYSICAL"
            self._emit("physical_grasp_success", cube.name)
        else:
            self.grasp_status = (
                "MAGNETIC FALLBACK" if self.grasp_mode == "auto" else "MAGNETIC"
            )
            self._emit("magnetic_attach", cube.name)
            self._emit("magnetic_fallback", cube.name)
        self.machine.advance()

    def _pose_for_state(self, state: TaskState, cube: CubeSpec) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.config["planning_frame"]
        cube_xyz = list(cube.position)
        bin_xyz = self._bin_drop_position(cube)
        offsets = self.config["offsets"]
        if state is TaskState.PRE_GRASP:
            xyz = [cube_xyz[0], cube_xyz[1], cube_xyz[2] + offsets["pre_grasp_z"]]
        elif state is TaskState.GRASP:
            xyz = [cube_xyz[0], cube_xyz[1], cube_xyz[2] + offsets["grasp_z"]]
        elif state is TaskState.LIFT:
            xyz = [cube_xyz[0], cube_xyz[1], cube_xyz[2] + offsets["lift_z"]]
        elif state is TaskState.TRANSIT:
            xyz = [bin_xyz[0], bin_xyz[1], bin_xyz[2] + offsets["place_z"]]
        elif state is TaskState.PLACE:
            xyz = bin_xyz
        elif state is TaskState.RETREAT:
            xyz = [bin_xyz[0], bin_xyz[1], bin_xyz[2] + offsets["retreat_z"]]
        else:
            raise RuntimeError(f"no Cartesian pose for {state.value}")
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, xyz)
        quaternion = self.config["tool_orientation_xyzw"]
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        return pose

    def _publish_world_cubes(self) -> None:
        from moveit_msgs.msg import PlanningScene

        scene = PlanningScene(is_diff=True)
        table = self.scene_config["table"]
        scene.world.collision_objects.append(
            _box_collision(
                "workstation_table",
                self.config["planning_frame"],
                table["center"],
                table["size"],
            )
        )
        bins = self.scene_config["bins"]
        inner = bins["inner_size"]
        wall = float(bins["wall_thickness"])
        base = float(bins["base_thickness"])
        for color, center_values in bins["centers"].items():
            x, y, z = map(float, center_values)
            dx, dy, dz = map(float, inner)
            pieces = {
                "base": ([x, y, z - dz / 2.0], [dx + 2 * wall, dy + 2 * wall, base]),
                "front": ([x - dx / 2.0 - wall / 2.0, y, z], [wall, dy + 2 * wall, dz]),
                "back": ([x + dx / 2.0 + wall / 2.0, y, z], [wall, dy + 2 * wall, dz]),
                "left": ([x, y + dy / 2.0 + wall / 2.0, z], [dx, wall, dz]),
                "right": ([x, y - dy / 2.0 - wall / 2.0, z], [dx, wall, dz]),
            }
            for piece, (center, size) in pieces.items():
                scene.world.collision_objects.append(
                    _box_collision(
                        f"{color}_bin_{piece}",
                        self.config["planning_frame"],
                        center,
                        size,
                    )
                )
        size = float(self.scene_config["cubes"]["size"])
        for cube in self.cubes:
            collision = CollisionObject()
            collision.header.frame_id = self.config["planning_frame"]
            collision.id = cube.name
            collision.primitives.append(
                SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[size, size, size])
            )
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = cube.position
            pose.orientation.w = 1.0
            collision.primitive_poses.append(pose)
            collision.operation = CollisionObject.ADD
            scene.world.collision_objects.append(collision)
        self._planning_scene_pub.publish(scene)

    def _attach_cube(self, cube: CubeSpec) -> None:
        from moveit_msgs.msg import PlanningScene

        attached = AttachedCollisionObject()
        attached.link_name = self.config["end_effector_link"]
        attached.object.header.frame_id = self.config["end_effector_link"]
        attached.object.id = cube.name
        size = float(self.scene_config["cubes"]["size"])
        attached.object.primitives.append(
            SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[size, size, size])
        )
        relative_pose = Pose()
        # The OpenArm v1 parallel gripper TCP is +0.0835 m along hand +Z.
        relative_pose.position.z = 0.0835
        relative_pose.orientation.w = 1.0
        attached.object.primitive_poses.append(relative_pose)
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = [
            "openarm_left_link7",
            "openarm_left_hand",
            "openarm_left_left_finger",
            "openarm_left_right_finger",
        ]
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)
        self._planning_scene_pub.publish(scene)

    def _remove_world_cube(self, cube: CubeSpec) -> None:
        from moveit_msgs.msg import PlanningScene

        collision = CollisionObject()
        collision.header.frame_id = self.config["planning_frame"]
        collision.id = cube.name
        collision.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(collision)
        self._planning_scene_pub.publish(scene)

    def _detach_cube(self, cube: CubeSpec) -> None:
        from moveit_msgs.msg import PlanningScene

        attached = AttachedCollisionObject()
        attached.link_name = self.config["end_effector_link"]
        attached.object.id = cube.name
        attached.object.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)
        self._planning_scene_pub.publish(scene)

    def _add_placed_cube(self, cube: CubeSpec) -> None:
        from moveit_msgs.msg import PlanningScene

        scene = PlanningScene(is_diff=True)
        size = float(self.scene_config["cubes"]["size"])
        bins = self.scene_config["bins"]
        bin_center = bins["centers"][cube.color]
        inner_height = float(bins["inner_size"][2])
        base_top = (
            float(bin_center[2])
            - inner_height / 2.0
            + float(bins["base_thickness"]) / 2.0
        )
        stack_level = self._stack_level(cube)
        cube_center = [
            float(bin_center[0]),
            float(bin_center[1]),
            base_top + size / 2.0 + stack_level * (size + 0.002),
        ]
        scene.world.collision_objects.append(
            _box_collision(
                cube.name,
                self.config["planning_frame"],
                cube_center,
                [size, size, size],
            )
        )
        self._planning_scene_pub.publish(scene)

    def _on_safety(self, message: String) -> None:
        state = SafetyState(message.data)
        if state in {SafetyState.PAUSE, SafetyState.EMERGENCY_STOP}:
            if self._move_goal_handle:
                self._move_goal_handle.cancel_goal_async()
            if self._trajectory_goal_handle:
                self._trajectory_goal_handle.cancel_goal_async()
        self.machine.set_safety(state)
        if state is SafetyState.RECOVER:
            self.machine.set_safety(SafetyState.SAFE)

    def _on_joint_state(self, message: JointState) -> None:
        required = set(self.robot_config["joint_names"]["left"])
        if required.issubset(message.name) and len(message.position) == len(message.name):
            self._joint_state_ready = True

    def _on_command(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "start":
            self.selected_only = False
            self._home_only = False
            self.started = True
        elif command.startswith("pick:"):
            cube_name = command.partition(":")[2]
            index = next(
                (index for index, cube in enumerate(self.cubes) if cube.name == cube_name),
                None,
            )
            if index is None:
                self._emit("command_rejected", "unknown_cube", cube_name)
            elif self.busy:
                self._emit("command_rejected", "robot_busy", cube_name)
            else:
                self.machine = TaskStateMachine()
                self.current_cube_index = index
                self.selected_only = True
                self._home_only = False
                self._planning_failures = 0
                self._fatal_error = False
                self._physical_grasp_attempt = 0
                self._current_grasp_assist = ""
                self.grasp_status = self.grasp_mode.upper()
                self.started = True
                self._emit("gui_pick_requested", cube_name, self.planner_mode)
        elif command == "home":
            if self.busy:
                self._emit("command_rejected", "robot_busy", "home")
            else:
                self.machine = TaskStateMachine()
                self.selected_only = False
                self._home_only = True
                self._planning_failures = 0
                self._fatal_error = False
                self.started = True
                self._emit("gui_home_requested")
        elif command == "pause":
            self.machine.set_safety(SafetyState.PAUSE)
        elif command == "resume":
            self.machine.set_safety(SafetyState.RECOVER)
            self.machine.set_safety(SafetyState.SAFE)
        elif command == "reset":
            if self._move_goal_handle:
                self._move_goal_handle.cancel_goal_async()
            if self._trajectory_goal_handle:
                self._trajectory_goal_handle.cancel_goal_async()
            cube = self._current_cube()
            if self._current_grasp_assist == "magnetic" and cube is not None:
                self._emit("magnetic_detach", cube.name)
            self.machine = TaskStateMachine()
            self.current_cube_index = 0
            self.selected_only = False
            self._home_only = False
            self._planning_failures = 0
            self._fatal_error = False
            self._physical_grasp_attempt = 0
            self._current_grasp_assist = ""
            self.grasp_status = self.grasp_mode.upper()
            self.busy = False
            self.started = False
            self._emit("gui_reset")

    def _on_planner_mode(self, message: String) -> None:
        mode = message.data.strip().lower()
        if mode not in {"moveit", "dynamic"}:
            self._emit("command_rejected", "unknown_planner", mode)
            return
        if self.busy:
            self._emit("command_rejected", "robot_busy", f"planner:{mode}")
            return
        self.planner_mode = mode
        self._emit("planner_mode", mode)

    def _on_grasp_mode(self, message: String) -> None:
        mode = message.data.strip().lower()
        if mode not in {"physical", "magnetic", "auto"}:
            self._emit("command_rejected", "unknown_grasp_mode", mode)
            return
        if self.busy or self.started:
            self._emit("command_rejected", "robot_busy", f"grasp:{mode}")
            return
        self.grasp_mode = mode
        self.grasp_status = mode.upper()
        self._emit("grasp_mode", mode)

    def _current_cube(self) -> CubeSpec | None:
        if self.current_cube_index >= len(self.cubes):
            return None
        return self.cubes[self.current_cube_index]

    def _stack_level(self, cube: CubeSpec) -> int:
        return sum(
            prior.color == cube.color
            for prior in self.cubes[: self.current_cube_index]
        )

    def _bin_drop_position(self, cube: CubeSpec) -> list[float]:
        position = [
            float(value) for value in self.config["bin_drop_positions"][cube.color]
        ]
        # Each same-color cube is released one cube layer higher. This avoids
        # inserting the fingers through an already sorted object while leaving
        # enough drop distance for the PhysX cube to settle naturally.
        position[2] += self._stack_level(cube) * (
            float(self.scene_config["cubes"]["size"]) + 0.010
        )
        return position

    def _emit(self, event: str, *values: str) -> None:
        self._event_pub.publish(String(data=",".join((event, *values))))


def _load_path(value: str) -> dict[str, Any]:
    path = Path(os.path.expandvars(value)).expanduser()
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _box_collision(
    object_id: str, frame_id: str, center: list[float], size: list[float]
) -> CollisionObject:
    collision = CollisionObject()
    collision.header.frame_id = frame_id
    collision.id = object_id
    collision.primitives.append(
        SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(value) for value in size])
    )
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, center)
    pose.orientation.w = 1.0
    collision.primitive_poses.append(pose)
    collision.operation = CollisionObject.ADD
    return collision


def main() -> None:
    rclpy.init()
    node = SortingTaskNode()
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
