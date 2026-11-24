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
        from rclpy.action import ActionServer, CancelResponse
        from rclpy.callback_groups import ReentrantCallbackGroup

        self.robot = robot
        self.node = node
        self.sim_time = 0.0
        self.safety_paused = False
        self._lock = threading.Lock()
        self._stopping = False
        self._active: dict[str, ActiveTrajectory] = {}
        self._active_grippers: dict[str, ActiveGripper] = {}
        # Isaac articulation reads and writes must stay on the simulation
        # thread.  ROS action callbacks consume this cache only.
        positions = robot.get_joint_positions()
        self._joint_positions = np.asarray(positions, dtype=float).copy()
        self._dof_names = list(robot.dof_names)
        callback_group = ReentrantCallbackGroup()
        self._servers = []
        for side in ("left", "right"):
            for action, name, callback in (
                (FollowJointTrajectory, side, self._execute_trajectory),
                (GripperCommand, f"{side}_gripper", self._execute_gripper),
            ):
                self._servers.append(
                    ActionServer(
                        node,
                        action,
                        action_names[name],
                        execute_callback=partial(callback, side=side),
                        cancel_callback=lambda _: CancelResponse.ACCEPT,
                        callback_group=callback_group,
                    )
                )

    def update(self, sim_time: float) -> None:
        self.sim_time = sim_time
        positions = self.robot.get_joint_positions()
        if positions is not None:
            self._joint_positions = np.asarray(positions, dtype=float).copy()
        with self._lock:
            for side, active in tuple(self._active.items()):
                self._update_trajectory(active, sim_time)
                if active.finished.is_set():
                    del self._active[side]
            for side, gripper in tuple(self._active_grippers.items()):
                self._update_gripper(gripper, sim_time)
                if gripper.finished.is_set():
                    del self._active_grippers[side]

    def _hold_if_stopped(self, active: ActiveTrajectory | ActiveGripper) -> bool:
        if not (self.safety_paused or active.goal_handle.is_cancel_requested):
            return False
        self._apply(active.joint_indices, self._joint_positions[active.joint_indices])
        active.aborted = self.safety_paused
        active.finished.set()
        return True

    def _update_trajectory(self, active: ActiveTrajectory, sim_time: float) -> None:
        if self._hold_if_stopped(active):
            return
        elapsed = sim_time - active.started_at
        targets, complete = self._interpolate(active, elapsed)
        self._apply(active.joint_indices, targets)
        if not complete:
            return
        error = float(np.max(np.abs(self._joint_positions[active.joint_indices] - targets)))
        if error <= self.JOINT_GOAL_TOLERANCE_RAD:
            self.node.get_logger().info(
                f"trajectory complete at sim_time={sim_time:.3f} max_joint_error={error:.5f}"
            )
            active.finished.set()
        elif elapsed >= _point_time(active.points[-1]) + self.GOAL_SETTLING_TIMEOUT_SEC:
            active.aborted = True
            active.goal_tolerance_violated = True
            self.node.get_logger().error(f"trajectory failed to settle: max_joint_error={error:.5f} rad")
            active.finished.set()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            for active in (*self._active.values(), *self._active_grippers.values()):
                active.aborted = True
                active.finished.set()
            self._active.clear()
            self._active_grippers.clear()

    def set_safety_state(self, state: str) -> None:
        self.safety_paused = state in {"PAUSE", "EMERGENCY_STOP"}

    def _execute_trajectory(self, goal_handle: Any, side: str) -> Any:
        from control_msgs.action import FollowJointTrajectory

        result = FollowJointTrajectory.Result()
        trajectory = goal_handle.request.trajectory
        names = self._dof_names
        try:
            if (
                not trajectory.joint_names
                or len(set(trajectory.joint_names)) != len(trajectory.joint_names)
                or any(not name.startswith(f"openarm_{side}_joint") for name in trajectory.joint_names)
            ):
                raise ValueError(f"expected unique {side} arm joints")
            indices = np.asarray([names.index(name) for name in trajectory.joint_names], dtype=int)
        except ValueError as error:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = str(error)
            return result
        times = [_point_time(point) for point in trajectory.points]
        if (
            not times
            or times[0] < 0.0
            or any(a >= b for a, b in zip(times, times[1:]))
            or any(
                len(point.positions) != len(indices) or not np.isfinite(point.positions).all()
                for point in trajectory.points
            )
        ):
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory has invalid joint positions or point times"
            return result
        active = ActiveTrajectory(
            goal_handle=goal_handle,
            joint_indices=indices,
            start_positions=self._joint_positions[indices].copy(),
            points=list(trajectory.points),
            started_at=self.sim_time,
            finished=threading.Event(),
        )
        duration = times[-1]
        self.node.get_logger().info(
            f"accepted trajectory joints={len(indices)} points={len(active.points)} "
            f"duration={duration:.3f}s start_sim_time={active.started_at:.3f}"
        )
        with self._lock:
            if self._stopping:
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                return result
            if previous := self._active.get(side):
                previous.aborted = True
                previous.finished.set()
            self._active[side] = active
        active.finished.wait()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "trajectory canceled"
        elif active.aborted:
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
        if not np.isfinite(target):
            goal_handle.abort()
            return result
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
            if self._stopping:
                goal_handle.abort()
                return result
            if previous := self._active_grippers.get(side):
                previous.aborted = True
                previous.finished.set()
            self._active_grippers[side] = active
        active.finished.wait()
        result.position = active.final_position
        result.reached_goal = active.reached_goal
        result.stalled = active.stalled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif active.aborted:
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return result

    def _update_gripper(self, active: ActiveGripper, sim_time: float) -> None:
        if self._hold_if_stopped(active):
            return
        targets = np.full(len(active.joint_indices), active.target, dtype=float)
        self._apply(active.joint_indices, targets)
        current = self._joint_positions[active.joint_indices]
        active.final_position = float(np.mean(current))
        if float(np.max(np.abs(current - active.target))) <= 0.002:
            active.reached_goal = True
            active.finished.set()
            return
        if float(np.max(np.abs(current - active.prior_positions))) > 2e-4:
            active.stalled_since = sim_time
            active.prior_positions = current.copy()
        # Closing on a cube is a valid stalled grasp rather than a false
        # target-reached report. Opening must actually reach its target.
        if active.target < 0.02 and sim_time - active.stalled_since >= 0.35:
            active.stalled = True
            active.finished.set()
            return
        if sim_time - active.started_at >= 2.0:
            active.stalled = True
            active.aborted = True
            active.finished.set()

    def _interpolate(self, active: ActiveTrajectory, elapsed: float) -> tuple[np.ndarray, bool]:
        prior_time = 0.0
        prior_positions = active.start_positions
        for point in active.points:
            point_time = _point_time(point)
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


def _point_time(point: Any) -> float:
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
