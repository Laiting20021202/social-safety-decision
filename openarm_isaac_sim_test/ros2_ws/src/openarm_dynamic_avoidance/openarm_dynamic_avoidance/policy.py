from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


# PAUSE cancels the invalid old path. REPLAN is intentionally executable: it
# denotes a newly collision-checked spatial path generated with the latest
# Planning Scene. Only emergency stop remains permanently hard-blocking.
BLOCKING_STATES = {"PAUSE", "EMERGENCY_STOP"}


def hold_recent_obstacle(
    obstacle_present: bool,
    last_seen_sec: float,
    now_sec: float,
    hold_sec: float,
) -> bool:
    """Keep a tracked obstacle through a short empty-mask inference gap."""

    return bool(
        obstacle_present
        and float(now_sec) - float(last_seen_sec) < max(float(hold_sec), 0.0)
    )


def obstacle_affects_segment(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    margin_m: float,
) -> bool:
    """Return whether a padded obstacle can affect the active TCP corridor."""

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((start, goal, center, size))).all():
        return True
    if np.any(size <= 0.0):
        return True
    half = size / 2.0 + max(float(margin_m), 0.0)
    lower, upper = center - half, center + half
    direction = goal - start
    near, far = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= 1e-9:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = float((lower[axis] - start[axis]) / direction[axis])
        second = float((upper[axis] - start[axis]) / direction[axis])
        if first > second:
            first, second = second, first
        near = max(near, first)
        far = min(far, second)
        if near > far:
            return False
    return True


def obstacle_affects_motion_corridor(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    margin_m: float,
) -> bool:
    """Check the TCP segment and its top-down articulated-arm corridor.

    A hand may not intersect the TCP's exact 3-D centerline while still
    crossing an elbow or forearm.  The XY fallback therefore remains useful,
    but only when the obstacle and active path also overlap vertically.  An
    unlimited top-down projection made a low hand cancel a high-arm motion.
    """

    if obstacle_affects_segment(
        start, goal, center, size, margin_m=margin_m
    ):
        return True
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((start, goal, center, size))).all():
        return True
    if np.any(size <= 0.0):
        return True
    margin = max(float(margin_m), 0.0)
    obstacle_lower_z = float(center[2] - size[2] / 2.0 - margin)
    obstacle_upper_z = float(center[2] + size[2] / 2.0 + margin)
    path_lower_z = float(min(start[2], goal[2]) - margin)
    path_upper_z = float(max(start[2], goal[2]) + margin)
    if obstacle_upper_z < path_lower_z or obstacle_lower_z > path_upper_z:
        return False
    half = size[:2] / 2.0 + margin
    lower, upper = center[:2] - half, center[:2] + half
    direction = goal[:2] - start[:2]
    near, far = 0.0, 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-9:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = float((lower[axis] - start[axis]) / direction[axis])
        second = float((upper[axis] - start[axis]) / direction[axis])
        if first > second:
            first, second = second, first
        near = max(near, first)
        far = min(far, second)
        if near > far:
            return False
    return True


def trajectory_blocked(safety_state: str, guarded_route_active: bool) -> bool:
    """Block every trajectory during PAUSE or EMERGENCY_STOP.

    A guarded bypass may be planned while paused, but it cannot execute until
    the supervisor advances to REPLAN.  A moving obstacle can invalidate even
    a previously checked bypass.
    """

    state = str(safety_state).strip().upper()
    _ = guarded_route_active
    return state in BLOCKING_STATES


def effective_velocity_scale(
    safety_state: str,
    safety_velocity_scale: float,
    guarded_route_velocity_scale: float,
) -> float:
    """Cap every dynamic-mode trajectory at the configured guarded speed.

    The safety supervisor may reduce the cap further in WARNING/REPLAN/RECOVER.
    SAFE must not silently restore full-speed execution, otherwise a short
    trajectory can finish before a moving obstacle is observed and replanned.
    """

    guarded = float(guarded_route_velocity_scale)
    safety = float(safety_velocity_scale)
    if not math.isfinite(guarded) or not 0.0 < guarded <= 1.0:
        raise ValueError("guarded_route_velocity_scale must be in (0, 1]")
    if not math.isfinite(safety) or not 0.0 <= safety <= 1.0:
        raise ValueError("safety_velocity_scale must be in [0, 1]")
    state = str(safety_state).strip().upper()
    if state in {"WARNING", "REPLAN", "RECOVER"}:
        return max(min(guarded, safety), 0.05)
    return guarded


def retime_trajectory(
    trajectory: Any,
    velocity_scale: float,
    current_positions: dict[str, float] | None = None,
) -> Any:
    """Return a monotonic, optionally rebased copy of a nominal trajectory."""

    scale = float(velocity_scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("velocity_scale must be finite and in (0, 1]")
    result = copy.deepcopy(trajectory)
    if not result.points:
        raise ValueError("trajectory contains no points")
    names = tuple(result.joint_names)
    if current_positions and all(name in current_positions for name in names):
        result.points[0].positions = [float(current_positions[name]) for name in names]
        result.points[0].velocities = [0.0] * len(names)
        result.points[0].accelerations = [0.0] * len(names)
    previous = -1.0
    for point in result.points:
        seconds = float(point.time_from_start.sec) + float(
            point.time_from_start.nanosec
        ) * 1e-9
        seconds /= scale
        if seconds <= previous:
            seconds = previous + 0.001
        whole = int(seconds)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int(round((seconds - whole) * 1e9))
        previous = seconds
        if point.velocities:
            point.velocities = [float(value) * scale for value in point.velocities]
        if point.accelerations:
            point.accelerations = [
                float(value) * scale * scale for value in point.accelerations
            ]
    return result
