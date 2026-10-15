from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def nearest_box_component(centers, sizes, arm_points):
    """Local reactive target; all other boxes remain in MoveIt's collision scene.

    Unioning an entire remembered scene makes free space between disconnected
    obstacles look occupied. Select the overlapping component nearest an arm.
    """
    centers, sizes = np.asarray(centers), np.asarray(sizes)
    points = np.asarray(arm_points).reshape(-1, 3)
    if not len(centers) or not len(points):
        return union_axis_aligned_boxes(centers, sizes)
    distance = np.linalg.norm(np.maximum(np.abs(centers[:, None] - points) - sizes[:, None] / 2, 0), axis=2).min(axis=1)
    active = {int(distance.argmin())}
    pending = list(active)
    while pending:
        index = pending.pop()
        touching = np.all(np.abs(centers - centers[index]) <= (sizes + sizes[index]) / 2 + .015, axis=1)
        for neighbor in np.flatnonzero(touching):
            if int(neighbor) not in active:
                active.add(int(neighbor)); pending.append(int(neighbor))
    indices = sorted(active)
    return union_axis_aligned_boxes(centers[indices], sizes[indices])


def active_escape_can_continue(
    safety_state: str,
    step_kind: str | None,
    busy: bool,
) -> bool:
    """Keep one checked upward escape across PAUSE -> REPLAN transitions."""

    return bool(
        busy
        and step_kind in {"safety_evade", "idle_evade"}
        and str(safety_state).strip().upper() in {"PAUSE", "REPLAN"}
    )


def point_to_box_distance(
    point: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
) -> float:
    """Return Euclidean distance from a point to an axis-aligned box."""

    value = np.asarray(point, dtype=float).reshape(3)
    box_center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(size, dtype=float).reshape(3) / 2.0
    if not np.isfinite(np.concatenate((value, box_center, half))).all():
        return float("inf")
    outside = np.maximum(np.abs(value - box_center) - half, 0.0)
    return float(np.linalg.norm(outside))


def reactive_escape_point(tcp, arm_points, center, size, *, side, maximum_step=.12):
    """Bounded local goal proposal using the closest measured arm/box pair.

    Inspired by DCP-RMP's local repulsion, not an implementation of its learned
    IMPACT policy. Preserve the measured tool orientation in the ROS adapter.
    """
    tcp, center, size = map(lambda v: np.asarray(v, dtype=float), (tcp, center, size))
    points = np.asarray(arm_points, dtype=float).reshape(-1, 3)
    if not len(points): points = tcp[None, :]
    closest = np.clip(points, center - size / 2, center + size / 2)
    offsets = points - closest
    index = int(np.argmin(np.linalg.norm(offsets, axis=1)))
    distance = float(np.linalg.norm(offsets[index]))
    away = offsets[index].copy()
    if np.linalg.norm(away) < 1e-6:
        away = points[index] - center
    if np.linalg.norm(away) < 1e-6:
        away = np.array([0., 1. if side == "left" else -1., .3])
    # The table is below the arm. Never create a downward retreat into it.
    away[2] = max(away[2], .03)
    away /= np.linalg.norm(away)
    step = float(np.clip(.22 - distance, .05, maximum_step))
    return tcp + away * step


def retime_local_path(positions, maximum_velocity=.4, maximum_acceleration=.8):
    """Time-scale a continuous local joint path; return times and velocities."""
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("local path needs at least two finite joint waypoints")
    dt = np.maximum(np.max(np.abs(np.diff(q, axis=0)), axis=1) / maximum_velocity, .05)
    times = np.r_[0., np.cumsum(dt)]
    for _ in range(4):
        velocity = np.gradient(q, times, axis=0)
        velocity[[0, -1]] = 0.
        acceleration = np.diff(velocity, axis=0) / np.diff(times)[:, None]
        factor = max(1., np.max(np.abs(velocity)) / maximum_velocity,
                     np.sqrt(np.max(np.abs(acceleration)) / maximum_acceleration))
        if factor <= 1.000001: break
        times *= factor * 1.01
    return times, velocity


def select_evading_sides(
    tcp_positions: Mapping[str, np.ndarray],
    center: np.ndarray,
    size: np.ndarray,
    *,
    trigger_distance_m: float,
) -> tuple[str, ...]:
    """Select every TCP near the hand, or the closest arm as a fallback.

    The safety supervisor evaluates complete link capsules, so a hand can be
    close to an elbow even when neither TCP lies inside the trigger radius.
    In that case moving the closest arm to its collision-checked Home pose is
    the deterministic safe fallback.
    """

    distances = {
        str(side): point_to_box_distance(point, center, size)
        for side, point in tcp_positions.items()
    }
    finite = {side: value for side, value in distances.items() if np.isfinite(value)}
    if not finite:
        return ()
    trigger = max(float(trigger_distance_m), 0.0)
    affected = tuple(
        side for side in ("left", "right") if finite.get(side, float("inf")) <= trigger
    )
    if affected:
        return affected
    return (min(finite, key=finite.get),)


def select_evading_sides_from_links(
    arm_points: Mapping[str, np.ndarray],
    center: np.ndarray,
    size: np.ndarray,
    *,
    trigger_distance_m: float,
) -> tuple[str, ...]:
    """Select arms using all sampled link positions, not only their TCPs."""

    distances: dict[str, float] = {}
    for side, values in arm_points.items():
        points = np.asarray(values, dtype=float)
        if points.size == 0 or points.size % 3 != 0:
            continue
        points = points.reshape(-1, 3)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            continue
        distances[str(side)] = min(
            point_to_box_distance(point, center, size) for point in finite
        )
    if not distances:
        return ()
    trigger = max(float(trigger_distance_m), 0.0)
    affected = tuple(
        side
        for side in ("left", "right")
        if distances.get(side, float("inf")) <= trigger
    )
    if affected:
        return affected
    return (min(distances, key=distances.get),)


def crisis_escape_point(
    tcp_position: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_size: np.ndarray,
    *,
    side: str,
    minimum_lift_m: float,
    horizontal_escape_m: float,
    vertical_clearance_m: float,
    maximum_tcp_z_m: float,
) -> np.ndarray:
    """Return an upward, obstacle-opposed TCP goal for a soft-stop escape.

    The goal is based only on the measured TCP and the live Planning Scene
    obstacle.  It never moves the arm toward the hand and never lowers a TCP
    that is already above the configured ceiling.
    """

    tcp = np.asarray(tcp_position, dtype=float).reshape(3)
    center = np.asarray(obstacle_center, dtype=float).reshape(3)
    size = np.asarray(obstacle_size, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((tcp, center, size))).all():
        raise ValueError("crisis escape geometry must be finite")
    if np.any(size <= 0.0):
        raise ValueError("crisis obstacle size must be positive")

    away = tcp[:2] - center[:2]
    norm = float(np.linalg.norm(away))
    if norm <= 1e-6:
        # Deterministic opposite lateral directions prevent both arms from
        # converging when the hand is centered between their TCPs.
        away = np.asarray([0.0, 1.0 if side == "left" else -1.0])
        norm = 1.0

    target = tcp.copy()
    target[:2] += away / norm * max(float(horizontal_escape_m), 0.0)
    obstacle_top = float(center[2] + size[2] / 2.0)
    wanted_z = max(
        float(tcp[2]) + max(float(minimum_lift_m), 0.0),
        obstacle_top + max(float(vertical_clearance_m), 0.0),
    )
    # Never turn a crisis lift into a downward motion when the TCP starts
    # above the normal workspace ceiling.
    target[2] = max(float(tcp[2]), min(wanted_z, float(maximum_tcp_z_m)))
    return target


def safety_evade_sequence(
    current_step: tuple[str, str],
    remaining_steps: list[tuple[str, str]],
    evading_sides: tuple[str, ...],
    *,
    target_kind: str,
    route_step_kinds: frozenset[str],
) -> list[tuple[str, str]]:
    """Preempt a moving-obstacle route with collision-checked Home retreats.

    Stale Cartesian bypass waypoints are discarded.  The interrupted target
    (or Home return) is preserved exactly once after the affected arms evade.
    """

    side, kind = current_step
    resume = (side, target_kind) if kind in route_step_kinds else current_step
    remaining = [
        step
        for step in remaining_steps
        if not (step[0] == side and step[1] in route_step_kinds)
    ]
    if not remaining or remaining[0] != resume:
        remaining.insert(0, resume)
    unique_sides = tuple(dict.fromkeys(evading_sides))
    return [(value, "safety_evade") for value in unique_sides] + remaining


def obstacle_has_withdrawn(
    trigger_center: np.ndarray | None,
    current_center: np.ndarray | None,
    *,
    minimum_motion_m: float,
) -> bool:
    """Return true only when the hand moved away, not when the robot did."""

    if current_center is None:
        return True
    if trigger_center is None:
        return False
    start = np.asarray(trigger_center, dtype=float).reshape(3)
    current = np.asarray(current_center, dtype=float).reshape(3)
    if not np.isfinite(np.concatenate((start, current))).all():
        return False
    return float(np.linalg.norm(current - start)) >= max(
        float(minimum_motion_m), 0.0
    )


def obstacle_is_clear_for_recovery(
    trigger_center: np.ndarray | None,
    current_center: np.ndarray | None,
    *,
    minimum_motion_m: float,
    current_distance_m: float,
    release_distance_m: float,
) -> bool:
    """Require obstacle withdrawal and measured robot-link clearance.

    Robot retreat alone can increase ``current_distance_m``.  Combining it
    with obstacle-center motion prevents that retreat from being mistaken for
    a clear scene.  A removed perception obstacle is considered clear because
    there is no current box to measure against.
    """

    if current_center is None:
        return True
    return bool(
        obstacle_has_withdrawn(
            trigger_center,
            current_center,
            minimum_motion_m=minimum_motion_m,
        )
        and np.isfinite(current_distance_m)
        and float(current_distance_m) >= max(float(release_distance_m), 0.0)
    )


def union_axis_aligned_boxes(
    centers: list[np.ndarray],
    sizes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return stable bounds covering every measured hand-cloud cluster.

    The perception bridge intentionally emits several compact boxes.  Using
    only primitive zero makes the apparent hand center jump between fingers
    or cloud partitions and repeatedly invalidates an otherwise safe route.
    """

    if len(centers) == 0 or len(centers) != len(sizes):
        return None
    center_values = [np.asarray(value, dtype=float).reshape(3) for value in centers]
    size_values = [np.asarray(value, dtype=float).reshape(3) for value in sizes]
    values = np.concatenate((*center_values, *size_values))
    if not np.isfinite(values).all() or any(np.any(value <= 0.0) for value in size_values):
        return None
    lower = np.min(
        np.stack([center - size / 2.0 for center, size in zip(center_values, size_values)]),
        axis=0,
    )
    upper = np.max(
        np.stack([center + size / 2.0 for center, size in zip(center_values, size_values)]),
        axis=0,
    )
    return (lower + upper) / 2.0, upper - lower
