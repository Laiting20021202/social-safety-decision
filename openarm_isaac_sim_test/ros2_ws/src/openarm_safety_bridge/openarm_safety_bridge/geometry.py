from __future__ import annotations

import numpy as np


def estimate_bounded_velocity(
    previous_center: np.ndarray | None,
    previous_stamp_sec: float | None,
    center: np.ndarray,
    stamp_sec: float,
    previous_velocity: np.ndarray,
    *,
    smoothing: float,
    maximum_speed_mps: float,
) -> np.ndarray:
    """Estimate obstacle velocity from acquisition stamps with bounded outliers."""

    current = np.asarray(center, dtype=float).reshape(3)
    old_velocity = np.asarray(previous_velocity, dtype=float).reshape(3)
    if previous_center is None or previous_stamp_sec is None:
        return np.zeros(3, dtype=float)
    dt = float(stamp_sec) - float(previous_stamp_sec)
    if not np.isfinite(dt) or dt <= 1e-4:
        return old_velocity.copy()
    raw = (current - np.asarray(previous_center, dtype=float).reshape(3)) / dt
    speed = float(np.linalg.norm(raw))
    limit = max(float(maximum_speed_mps), 0.0)
    if not np.isfinite(raw).all():
        return old_velocity.copy()
    if limit > 0.0 and speed > limit:
        raw *= limit / speed
    alpha = float(np.clip(smoothing, 0.0, 1.0))
    return (1.0 - alpha) * old_velocity + alpha * raw


def limit_cloud_center_motion(
    points: np.ndarray,
    previous_center: np.ndarray | None,
    previous_stamp_sec: float | None,
    stamp_sec: float,
    *,
    maximum_speed_mps: float,
    slack_m: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Bound neural correction jumps while retaining fresh RGB-D geometry.

    A new neural mask can occasionally lock onto an OpenArm link for one
    frame.  Its cloud then teleports several decimetres even though the
    kinematic hand can only move continuously.  Translate that fresh cloud
    back to the largest physically reachable center for the elapsed sensor
    time.  Shape and per-frame depth still come from the current RGB-D image.
    """

    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    center = np.median(cloud, axis=0)
    if previous_center is None or previous_stamp_sec is None:
        return cloud.copy(), center, False
    dt = float(stamp_sec) - float(previous_stamp_sec)
    if not np.isfinite(dt) or dt <= 1e-4:
        return cloud.copy(), center, False
    delta = center - np.asarray(previous_center, dtype=float).reshape(3)
    travel = float(np.linalg.norm(delta))
    allowed = max(float(maximum_speed_mps), 0.0) * dt + max(float(slack_m), 0.0)
    if not np.isfinite(travel) or travel <= allowed or travel <= 1e-12:
        return cloud.copy(), center, False
    bounded_center = np.asarray(previous_center, dtype=float).reshape(3) + (
        delta * (allowed / travel)
    )
    return cloud + (bounded_center - center), bounded_center, True


def swept_axis_aligned_box(
    center: np.ndarray,
    size: np.ndarray,
    velocity: np.ndarray,
    horizon_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an AABB covering the measured box and its predicted position."""

    origin = np.asarray(center, dtype=float).reshape(3)
    dimensions = np.asarray(size, dtype=float).reshape(3)
    displacement = np.asarray(velocity, dtype=float).reshape(3) * max(
        float(horizon_sec), 0.0
    )
    return origin + displacement / 2.0, dimensions + np.abs(displacement)


def clustered_swept_boxes(
    points: np.ndarray,
    velocity: np.ndarray,
    horizon_sec: float,
    *,
    padding_m: float,
    maximum_boxes: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate a hand cloud with compact boxes instead of one empty AABB.

    The hand mask is long and non-convex.  A single axis-aligned box fills the
    empty space between fingers/palm extremes and can engulf a robot link that
    is still far from every measured point.  Quantile slices along the cloud's
    longest axis preserve the observed geometry without simulator truth.
    """

    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        return []
    count = max(1, min(int(maximum_boxes), len(cloud)))
    global_lower = np.quantile(cloud, 0.02, axis=0)
    global_upper = np.quantile(cloud, 0.98, axis=0)
    axis = int(np.argmax(global_upper - global_lower))
    ordered = cloud[np.argsort(cloud[:, axis])]
    padding = max(float(padding_m), 0.0)
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in np.array_split(ordered, count):
        if not len(chunk):
            continue
        lower = np.quantile(chunk, 0.02, axis=0)
        upper = np.quantile(chunk, 0.98, axis=0)
        size = np.maximum(upper - lower, 0.025)
        center = (lower + upper) / 2.0
        swept_center, swept_size = swept_axis_aligned_box(
            center, size, velocity, horizon_sec
        )
        boxes.append((swept_center, swept_size + 2.0 * padding))
    return boxes


def minimum_cloud_to_capsules_distance(
    points: np.ndarray,
    chains: list[np.ndarray],
    *,
    capsule_radius_m: float,
    distance_quantile: float = 0.0,
) -> float:
    """Minimum surface clearance from cloud points to articulated link segments."""

    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    radius = max(float(capsule_radius_m), 0.0)
    quantile = float(np.clip(distance_quantile, 0.0, 0.25))
    if not len(cloud) or not chains:
        return float("inf")
    best = float("inf")
    for values in chains:
        chain = np.asarray(values, dtype=float).reshape(-1, 3)
        chain = chain[np.isfinite(chain).all(axis=1)]
        if not len(chain):
            continue
        if len(chain) == 1:
            distances = np.linalg.norm(cloud - chain[0], axis=1)
            best = min(best, float(np.quantile(distances, quantile)))
            continue
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            direction = end - start
            squared_length = float(np.dot(direction, direction))
            if squared_length <= 1e-12:
                distances = np.linalg.norm(cloud - start, axis=1)
            else:
                projection = np.clip(
                    ((cloud - start) @ direction) / squared_length, 0.0, 1.0
                )
                closest = start + projection[:, None] * direction
                distances = np.linalg.norm(cloud - closest, axis=1)
            best = min(best, float(np.quantile(distances, quantile)))
    return max(best - radius, 0.0)
