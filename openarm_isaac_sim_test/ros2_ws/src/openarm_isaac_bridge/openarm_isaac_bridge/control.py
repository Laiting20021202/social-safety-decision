from __future__ import annotations

import threading
from functools import partial
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ActiveTrajectory:
    goal_handle: Any
    joint_indices: np.ndarray
    start_positions: np.ndarray
    points: list[Any]
    started_at: float
    finished: threading.Event
    aborted: bool = False
    goal_tolerance_violated: bool = False


@dataclass
class ActiveGripper:
    goal_handle: Any
    joint_indices: np.ndarray
    target: float
    started_at: float
    prior_positions: np.ndarray
    stalled_since: float
    finished: threading.Event
    final_position: float = 0.0
    reached_goal: bool = False
    stalled: bool = False
    aborted: bool = False


class TrajectoryController:
    JOINT_GOAL_TOLERANCE_RAD = 0.015
    GOAL_SETTLING_TIMEOUT_SEC = 2.0

    def __init__(self, robot: Any, node: Any, action_names: dict[str, str]) -> None:
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from rclpy.action import ActionServer

        self.robot = robot
        self.node = node
        self.sim_time = 0.0
        self.safety_paused = False
        self._lock = threading.Lock()
        self._active: ActiveTrajectory | None = None
        self._active_gripper: ActiveGripper | None = None
        # Isaac articulation reads and writes must stay on the simulation
        # thread.  ROS action callbacks consume this cache only.
        positions = robot.get_joint_positions()
        self._joint_positions = np.asarray(positions, dtype=float).copy()
        self._dof_names = list(robot.dof_names)
        self._servers = [
            ActionServer(
                node,
                FollowJointTrajectory,
                action_names["left"],
                execute_callback=self._execute_trajectory,
            ),
            ActionServer(
                node,
                FollowJointTrajectory,
                action_names["right"],
                execute_callback=self._execute_trajectory,
            ),
            ActionServer(
                node,
                GripperCommand,
                action_names["left_gripper"],
                execute_callback=partial(self._execute_gripper, side="left"),
            ),
            ActionServer(
                node,
                GripperCommand,
                action_names["right_gripper"],
                execute_callback=partial(self._execute_gripper, side="right"),
            ),
        ]

    def update(self, sim_time: float) -> None:
        self.sim_time = sim_time
        positions = self.robot.get_joint_positions()
        if positions is not None:
            self._joint_positions = np.asarray(positions, dtype=float).copy()
        with self._lock:
            active = self._active
            if active is not None:
                if self.safety_paused:
                    active.aborted = True
                    active.finished.set()
                    self._active = None
                else:
                    elapsed = sim_time - active.started_at
                    targets, complete = self._interpolate(active, elapsed)
                    self._apply(active.joint_indices, targets)
                    if complete:
                        error = float(
                            np.max(
                                np.abs(
                                    self._joint_positions[active.joint_indices] - targets
                                )
                            )
                        )
                        final_point = active.points[-1]
                        final_time = (
                            final_point.time_from_start.sec
                            + final_point.time_from_start.nanosec * 1e-9
                        )
                        if error <= self.JOINT_GOAL_TOLERANCE_RAD:
                            self.node.get_logger().info(
                                "trajectory complete at "
                                f"sim_time={sim_time:.3f} max_joint_error={error:.5f}"
                            )
                            active.finished.set()
                            self._active = None
                        elif elapsed >= final_time + self.GOAL_SETTLING_TIMEOUT_SEC:
                            active.aborted = True
                            active.goal_tolerance_violated = True
                            self.node.get_logger().error(
                                "trajectory failed to settle: "
                                f"max_joint_error={error:.5f} rad"
                            )
                            active.finished.set()
                            self._active = None
            gripper = self._active_gripper
            if gripper is not None:
                self._update_gripper(gripper, sim_time)

    def set_safety_state(self, state: str) -> None:
        self.safety_paused = state in {"PAUSE", "EMERGENCY_STOP"}

    def _execute_trajectory(self, goal_handle: Any) -> Any:
        from control_msgs.action import FollowJointTrajectory

        result = FollowJointTrajectory.Result()
        trajectory = goal_handle.request.trajectory
        names = self._dof_names
        try:
            indices = np.asarray([names.index(name) for name in trajectory.joint_names], dtype=int)
        except ValueError as error:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = str(error)
            return result
        if not trajectory.points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory contains no points"
            return result
        active = ActiveTrajectory(
            goal_handle=goal_handle,
            joint_indices=indices,
            start_positions=self._joint_positions[indices].copy(),
            points=list(trajectory.points),
            started_at=self.sim_time,
            finished=threading.Event(),
        )
        final_point = active.points[-1]
        duration = final_point.time_from_start.sec + final_point.time_from_start.nanosec * 1e-9
        self.node.get_logger().info(
            f"accepted trajectory joints={len(indices)} points={len(active.points)} "
            f"duration={duration:.3f}s start_sim_time={active.started_at:.3f}"
        )
        with self._lock:
            if self._active is not None:
                self._active.aborted = True
                self._active.finished.set()
            self._active = active
        while not active.finished.wait(timeout=0.01):
            if goal_handle.is_cancel_requested:
                active.aborted = True
                with self._lock:
                    if self._active is active:
                        self._active = None
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "trajectory canceled"
                return result
        if active.aborted:
            goal_handle.abort()
            if active.goal_tolerance_violated:
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                result.error_string = "Isaac articulation did not settle at the final target"
            else:
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                result.error_string = "trajectory stopped by OpenArm safety supervisor"
        else:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _execute_gripper(self, goal_handle: Any, side: str) -> Any:
        from control_msgs.action import GripperCommand

        result = GripperCommand.Result()
        candidates = [
            index
            for index, name in enumerate(self._dof_names)
            if name.startswith(f"openarm_{side}_finger_joint")
        ]
        if len(candidates) != 2:
            goal_handle.abort()
            result.reached_goal = False
            return result
        target = float(goal_handle.request.command.position)
        indices = np.asarray(candidates, dtype=int)
        current = self._joint_positions[indices].copy()
        active = ActiveGripper(
            goal_handle=goal_handle,
            joint_indices=indices,
            target=target,
            started_at=self.sim_time,
            prior_positions=current,
            stalled_since=self.sim_time,
            finished=threading.Event(),
            final_position=float(np.mean(current)),
        )
        self.node.get_logger().info(
            f"accepted {side} gripper target={target:.4f} "
            f"start_sim_time={active.started_at:.3f}"
        )
        with self._lock:
            if self._active_gripper is not None:
                self._active_gripper.aborted = True
                self._active_gripper.finished.set()
            self._active_gripper = active
        while not active.finished.wait(timeout=0.01):
            if goal_handle.is_cancel_requested:
                active.aborted = True
                with self._lock:
                    if self._active_gripper is active:
                        self._active_gripper = None
                goal_handle.canceled()
                return result
        result.position = active.final_position
        result.reached_goal = active.reached_goal
        result.stalled = active.stalled
        if active.aborted:
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return result

    def _update_gripper(self, active: ActiveGripper, sim_time: float) -> None:
        if self.safety_paused:
            active.aborted = True
            active.finished.set()
            self._active_gripper = None
            return
        targets = np.full(len(active.joint_indices), active.target, dtype=float)
        self._apply(active.joint_indices, targets)
        current = self._joint_positions[active.joint_indices]
        active.final_position = float(np.mean(current))
        if float(np.max(np.abs(current - active.target))) <= 0.002:
            active.reached_goal = True
            active.finished.set()
            self._active_gripper = None
            return
        if float(np.max(np.abs(current - active.prior_positions))) > 2e-4:
            active.stalled_since = sim_time
            active.prior_positions = current.copy()
        # Closing on a cube is a valid stalled grasp rather than a false
        # target-reached report. Opening must actually reach its target.
        if active.target < 0.02 and sim_time - active.stalled_since >= 0.35:
            active.stalled = True
            active.finished.set()
            self._active_gripper = None
            return
        if sim_time - active.started_at >= 2.0:
            active.stalled = True
            active.aborted = True
            active.finished.set()
            self._active_gripper = None

    def _interpolate(self, active: ActiveTrajectory, elapsed: float) -> tuple[np.ndarray, bool]:
        prior_time = 0.0
        prior_positions = active.start_positions
        for point in active.points:
            point_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            current = np.asarray(point.positions, dtype=float)
            if elapsed <= point_time:
                span = max(point_time - prior_time, 1e-9)
                ratio = np.clip((elapsed - prior_time) / span, 0.0, 1.0)
                return prior_positions + ratio * (current - prior_positions), False
            prior_time = point_time
            prior_positions = current
        return prior_positions, True

    def _apply(self, indices: np.ndarray, positions: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=positions, joint_indices=indices)
        )
