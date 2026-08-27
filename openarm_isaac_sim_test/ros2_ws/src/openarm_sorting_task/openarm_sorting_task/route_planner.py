from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def waiting_path_should_retry(
    safety_state: str,
    current_distance_m: float,
    release_distance_m: float,
    *,
    coarse_path_blocked: bool,
) -> bool:
    """Allow a fresh MoveIt attempt once measured link clearance is safe.

    The simple TCP-to-box corridor is intentionally conservative. It may
    remain blocked after the hand has moved far enough for MoveIt to find a
    collision-free joint-space path, so it must not permanently gate retries.
    """

    if not coarse_path_blocked:
        return True
    return bool(
        str(safety_state).strip().upper() in {"SAFE", "RECOVER"}
        and np.isfinite(current_distance_m)
        and float(current_distance_m) >= max(float(release_distance_m), 0.0)
    )


@dataclass(frozen=True)
class UnderRoute:
    """Two Cartesian guide points that carry a TCP around an obstacle."""

    entry: np.ndarray
    exit: np.ndarray
    strategy: str = "under"


def touch_then_home_sequence(
    sides: tuple[str, ...], target_kind: str = "model_target"
) -> list[tuple[str, str]]:
    """Build a zero-dwell target-touch sequence with a guaranteed Home leg.

    Each arm returns Home immediately after its target trajectory succeeds.
    Keeping the Home leg adjacent to its target also makes the dual-arm command
    deterministic and prevents one arm from waiting at the marker while the
    other arm is still planning.
    """

    if target_kind not in {"model_target", "explicit_target"}:
        raise ValueError(f"unsupported target kind: {target_kind}")
    sequence: list[tuple[str, str]] = []
    for side in sides:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported OpenArm side: {side}")
        sequence.extend(((side, target_kind), (side, "home")))
    return sequence


def target_hold_sequence(
    sides: tuple[str, ...], target_kind: str = "model_target"
) -> list[tuple[str, str]]:
    """Build target-only steps; Home is an explicit operator command."""

    if target_kind not in {"model_target", "explicit_target"}:
        raise ValueError(f"unsupported target kind: {target_kind}")
    sequence: list[tuple[str, str]] = []
    for side in sides:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported OpenArm side: {side}")
        sequence.append((side, target_kind))
    return sequence


def preserve_home_after_side_failure(
    sequence: list[tuple[str, str]], side: str
) -> list[tuple[str, str]]:
    """Drop stale route/target legs while retaining the failed arm's Home leg."""

    return [
        queued
        for queued in sequence
        if queued[0] != side or queued[1] == "home"
    ]


def target_contact_point(
    marker_center: np.ndarray,
    tcp_position: np.ndarray,
    marker_radius_m: float,
) -> np.ndarray:
    """Return the marker surface point facing the current TCP.

    The Gazebo cubes are visible, collision-free goal markers. Requiring the
    TCP to occupy their geometric center can put an otherwise touchable marker
    just outside the arm's IK workspace. The physical task is fingertip
    contact, so the near surface is the correct Cartesian goal.
    """

    center = np.asarray(marker_center, dtype=float).reshape(3)
    tcp = np.asarray(tcp_position, dtype=float).reshape(3)
    radius = max(float(marker_radius_m), 0.0)
    direction = tcp - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9 or radius == 0.0:
        return center.copy()
    return center + direction / norm * radius


def progressive_pose_tolerance(
    base_m: float,
    failed_attempts: int,
    maximum_m: float,
) -> float:
    """Widen a boundary IK goal gradually while preserving first-try precision."""

    base = max(float(base_m), 1e-4)
    maximum = max(float(maximum_m), base)
    failures = max(int(failed_attempts), 0)
    return min(base * (1.0 + 0.5 * failures), maximum)


def guarded_route_needs_restart(
    safety_state: str,
    guarded_route_active: bool,
    planner_busy: bool,
) -> bool:
    """Return whether REPLAN must restart a PAUSE-cancelled route leg."""

    return bool(
        str(safety_state).strip().upper() == "REPLAN"
        and guarded_route_active
        and not planner_busy
    )


def segment_intersects_box(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    margin_m: float = 0.0,
) -> bool:
    """Return whether a finite 3-D segment intersects an axis-aligned box."""

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(size, dtype=float).reshape(3) / 2.0 + max(float(margin_m), 0.0)
    lower, upper = center - half, center + half
    direction = goal - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) <= 1e-9:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / direction[axis]
        second = (upper[axis] - start[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        t_min = max(t_min, float(first))
        t_max = min(t_max, float(second))
        if t_min > t_max:
            return False
    return True


def under_route_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    collision_margin_m: float,
    horizontal_clearance_m: float,
    vertical_clearance_m: float,
    minimum_tcp_z_m: float,
    entry_horizontal_clearance_m: float | None = None,
    exit_horizontal_clearance_m: float | None = None,
) -> UnderRoute | None:
    """Build entry/exit points below a blocking box, or report infeasibility.

    This is a deterministic guide for MoveIt/OMPL, not a learned policy. MoveIt
    still validates the complete robot collision geometry for both legs.
    """

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((start, goal, center, size))).all():
        return None
    if np.any(size <= 0.0):
        return None

    horizontal = goal[:2] - start[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-6:
        return None
    # A TCP centerline may pass above a hand while the elbow/forearm still
    # collides. Treat an obstacle that blocks the top-down travel corridor as
    # a reason to prefer the requested under route as well. MoveIt remains the
    # authority that validates the full robot geometry for every segment.
    margin = max(float(collision_margin_m), 0.0)
    intersects_3d = segment_intersects_box(
        start, goal, center, size, margin_m=margin
    )
    intersects_xy = _segment_intersects_xy_box(
        start[:2], goal[:2], center[:2], size[:2] / 2.0 + margin
    )
    if not intersects_3d and not intersects_xy:
        return None
    direction = horizontal / horizontal_norm
    half = size[:2] / 2.0 + margin
    projected_half_extent = float(np.dot(np.abs(direction), half))
    default_clearance = max(float(horizontal_clearance_m), 0.0)
    entry_offset = projected_half_extent + max(
        float(
            default_clearance
            if entry_horizontal_clearance_m is None
            else entry_horizontal_clearance_m
        ),
        0.0,
    )
    exit_offset = projected_half_extent + max(
        float(
            default_clearance
            if exit_horizontal_clearance_m is None
            else exit_horizontal_clearance_m
        ),
        0.0,
    )
    under_z = float(center[2] - size[2] / 2.0 - max(float(vertical_clearance_m), 0.0))
    if under_z < float(minimum_tcp_z_m):
        return None

    entry = np.array(
        [
            center[0] - direction[0] * entry_offset,
            center[1] - direction[1] * entry_offset,
            under_z,
        ],
        dtype=float,
    )
    padded_half = size / 2.0 + margin
    goal_is_below_box = bool(
        np.all(np.abs(goal[:2] - center[:2]) <= padded_half[:2])
        and goal[2] < center[2] - padded_half[2]
    )
    if goal_is_below_box:
        # The destination itself is underneath the hand. Crossing to the far
        # side and then reversing is longer and can exceed the arm workspace.
        # Move to a collision-cleared point directly above the destination,
        # then let the final short segment descend onto the marker.
        exit = np.array([goal[0], goal[1], under_z], dtype=float)
    else:
        exit = np.array(
            [
                center[0] + direction[0] * exit_offset,
                center[1] + direction[1] * exit_offset,
                under_z,
            ],
            dtype=float,
        )
    # Preserve travel direction if the geometric start/goal order is reversed
    # relative to the obstacle projection.
    if (
        not goal_is_below_box
        and np.dot(entry[:2] - start[:2], direction)
        > np.dot(exit[:2] - start[:2], direction)
    ):
        entry, exit = exit, entry
    return UnderRoute(entry=entry, exit=exit, strategy="under")


def avoidance_route_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    collision_margin_m: float,
    horizontal_clearance_m: float,
    vertical_clearance_m: float,
    minimum_tcp_z_m: float,
    maximum_tcp_z_m: float,
    entry_horizontal_clearance_m: float | None = None,
    exit_horizontal_clearance_m: float | None = None,
    maximum_backtrack_m: float = 0.02,
    force_local_detour: bool = False,
    forced_local_clearance_m: float | None = None,
) -> UnderRoute | None:
    """Choose a deterministic below/above/side guide around a blocking box.

    The earlier demo only generated a below-obstacle guide.  That deadlocked
    whenever the table removed the lower clearance.  Prefer that requested
    path when it is physically possible, then use an upper guide and finally
    a lateral guide.  MoveIt remains responsible for validating the complete
    robot collision geometry for every segment.
    """

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((start, goal, center, size))).all():
        return None
    if np.any(size <= 0.0):
        return None

    horizontal = goal[:2] - start[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-6:
        return None
    direction = horizontal / horizontal_norm
    margin = max(float(collision_margin_m), 0.0)
    half = size / 2.0 + margin
    if force_local_detour:
        # After a moving obstacle has caused PAUSE, never switch back to an
        # under-hand route on the next cloud refresh.  The hand can overtake a
        # slow descending arm before the new trajectory finishes.
        return _forced_local_side_route(
            start,
            goal,
            center,
            half,
            direction,
            horizontal_clearance_m=(
                horizontal_clearance_m
                if forced_local_clearance_m is None
                else forced_local_clearance_m
            ),
        )

    under = under_route_waypoints(
        start,
        goal,
        center,
        size,
        collision_margin_m=collision_margin_m,
        horizontal_clearance_m=horizontal_clearance_m,
        vertical_clearance_m=vertical_clearance_m,
        minimum_tcp_z_m=minimum_tcp_z_m,
        entry_horizontal_clearance_m=entry_horizontal_clearance_m,
        exit_horizontal_clearance_m=exit_horizontal_clearance_m,
    )
    if under is not None:
        return _limit_route_backtrack(
            under,
            start,
            goal,
            center,
            size,
            collision_margin_m=collision_margin_m,
            horizontal_clearance_m=horizontal_clearance_m,
            maximum_backtrack_m=maximum_backtrack_m,
        )

    intersects_xy = _segment_intersects_xy_box(
        start[:2], goal[:2], center[:2], half[:2]
    )
    if not intersects_xy:
        return None
    projected_half_extent = float(np.dot(np.abs(direction), half[:2]))
    default_clearance = max(float(horizontal_clearance_m), 0.0)
    entry_clearance = max(
        default_clearance
        if entry_horizontal_clearance_m is None
        else float(entry_horizontal_clearance_m),
        0.0,
    )
    exit_clearance = max(
        default_clearance
        if exit_horizontal_clearance_m is None
        else float(exit_horizontal_clearance_m),
        0.0,
    )
    entry_offset = projected_half_extent + entry_clearance
    exit_offset = projected_half_extent + exit_clearance

    over_z = float(center[2] + half[2] + max(float(vertical_clearance_m), 0.0))
    if over_z <= float(maximum_tcp_z_m):
        entry = np.array(
            [
                center[0] - direction[0] * entry_offset,
                center[1] - direction[1] * entry_offset,
                over_z,
            ],
            dtype=float,
        )
        exit = np.array(
            [
                center[0] + direction[0] * exit_offset,
                center[1] + direction[1] * exit_offset,
                over_z,
            ],
            dtype=float,
        )
        return _limit_route_backtrack(
            UnderRoute(entry=entry, exit=exit, strategy="over"),
            start,
            goal,
            center,
            size,
            collision_margin_m=collision_margin_m,
            horizontal_clearance_m=horizontal_clearance_m,
            maximum_backtrack_m=maximum_backtrack_m,
        )

    # Last resort: skirt the shorter horizontal side while maintaining a
    # height already known to be reachable at either end of the request.
    normal = np.array([-direction[1], direction[0]], dtype=float)
    side_extent = float(np.dot(np.abs(normal), half[:2])) + default_clearance
    candidates = []
    route_z = float(max(goal[2], min(start[2], center[2])))
    for sign in (-1.0, 1.0):
        offset = normal * side_extent * sign
        entry = np.array(
            [
                center[0] - direction[0] * entry_offset + offset[0],
                center[1] - direction[1] * entry_offset + offset[1],
                route_z,
            ],
            dtype=float,
        )
        exit = np.array(
            [
                center[0] + direction[0] * exit_offset + offset[0],
                center[1] + direction[1] * exit_offset + offset[1],
                route_z,
            ],
            dtype=float,
        )
        cost = float(np.linalg.norm(entry - start) + np.linalg.norm(exit - goal))
        candidates.append((cost, entry, exit))
    _, entry, exit = min(candidates, key=lambda candidate: candidate[0])
    return _limit_route_backtrack(
        UnderRoute(entry=entry, exit=exit, strategy="side"),
        start,
        goal,
        center,
        size,
        collision_margin_m=collision_margin_m,
        horizontal_clearance_m=horizontal_clearance_m,
        maximum_backtrack_m=maximum_backtrack_m,
    )


def _forced_local_side_route(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    padded_half: np.ndarray,
    direction: np.ndarray,
    *,
    horizontal_clearance_m: float,
) -> UnderRoute:
    """Create a local sidestep when a robot link, not the TCP line, is blocked."""

    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    obstacle_side = float(np.dot(center[:2] - start[:2], normal))
    sign = -1.0 if obstacle_side >= 0.0 else 1.0
    side_step = float(np.dot(np.abs(normal), padded_half[:2])) + max(
        float(horizontal_clearance_m), 0.0
    )
    lateral = normal * side_step * sign
    obstacle_progress = float(np.dot(center[:2] - start[:2], direction))
    forward_extent = float(np.dot(np.abs(direction), padded_half[:2]))
    goal_distance = float(np.linalg.norm(goal[:2] - start[:2]))
    exit_progress = min(
        goal_distance,
        max(0.06, obstacle_progress + forward_extent + horizontal_clearance_m),
    )
    entry = start.copy()
    entry[:2] += lateral
    exit = start.copy()
    exit[:2] += direction * exit_progress + lateral
    return UnderRoute(entry=entry, exit=exit, strategy="side_forced_progress")


def route_progress(
    start: np.ndarray,
    goal: np.ndarray,
    waypoint: np.ndarray,
) -> float:
    """Return signed horizontal progress along the current TCP-to-goal line."""

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    waypoint = np.asarray(waypoint, dtype=float).reshape(3)
    direction = goal[:2] - start[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return 0.0
    return float(np.dot(waypoint[:2] - start[:2], direction / norm))


def _limit_route_backtrack(
    route: UnderRoute,
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    *,
    collision_margin_m: float,
    horizontal_clearance_m: float,
    maximum_backtrack_m: float,
) -> UnderRoute:
    """Replace a retreating entry with a local sideways entry.

    A moving hand can overtake the TCP while a trajectory is being cancelled.
    Reusing the geometric near-side waypoint then sends the arm backwards and
    makes it lose all target progress.  Keep the new entry at the TCP's current
    longitudinal station and move only toward the closest free side of the
    padded hand box.  MoveIt still collision-checks the complete robot before
    this deterministic guide is executed.
    """

    maximum_backtrack = max(float(maximum_backtrack_m), 0.0)
    if route_progress(start, goal, route.entry) >= -maximum_backtrack:
        return route

    horizontal = goal[:2] - start[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-9:
        return route
    direction = horizontal / horizontal_norm
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    half = size[:2] / 2.0 + max(float(collision_margin_m), 0.0)
    side_extent = float(np.dot(np.abs(normal), half)) + max(
        float(horizontal_clearance_m), 0.0
    )
    start_side = float(np.dot(start[:2] - center[:2], normal))
    desired_side = min(
        (-side_extent, side_extent), key=lambda value: abs(value - start_side)
    )
    entry = route.entry.copy()
    entry[:2] = start[:2] + normal * (desired_side - start_side)

    exit = route.exit.copy()
    # A side route must stay on one side until it clears the obstacle.  Routes
    # above/below it are already separated in Z and may keep their original
    # exit, which is usually closer to the requested marker.
    if route.strategy == "side":
        exit_side = float(np.dot(exit[:2] - center[:2], normal))
        exit[:2] += normal * (desired_side - exit_side)
    return UnderRoute(
        entry=entry,
        exit=exit,
        strategy=f"{route.strategy}_progress",
    )


def _segment_intersects_xy_box(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
) -> bool:
    lower, upper = center - half, center + half
    direction = goal - start
    t_min, t_max = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) <= 1e-9:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / direction[axis]
        second = (upper[axis] - start[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        t_min = max(t_min, float(first))
        t_max = min(t_max, float(second))
        if t_min > t_max:
            return False
    return True
