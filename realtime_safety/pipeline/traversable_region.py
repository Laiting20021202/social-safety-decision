from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from realtime_safety.pipeline.ground_plane import GroundPlane
from realtime_safety.types import DangerZone, Track3DState


@dataclass(slots=True)
class TraversableRegion:
    points: np.ndarray
    polygon_xyz: np.ndarray
    confidence: float


def compute_traversable_region(
    cloud_points: np.ndarray,
    ground: GroundPlane | None,
    tracks: list[Track3DState],
    zones: list[DangerZone],
    clearance: float = 0.45,
) -> TraversableRegion:
    if ground is None:
        return TraversableRegion(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0.0)
    points = np.asarray(cloud_points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 1] > 0.0) & (points[:, 1] < 12.0) & (np.abs(points[:, 0]) < 6.0)
    distance = np.abs(points @ ground.coefficients[:3] + ground.coefficients[3])
    valid &= distance < 0.12
    ground_points = points[valid]
    for track in tracks:
        planar_distance = np.linalg.norm(ground_points[:, :2] - track.position_xyz[None, :2], axis=1)
        ground_points = ground_points[planar_distance > track.radius + clearance]
    for zone in zones:
        for position, radius in zip(zone.predicted_positions[::2], zone.radii[::2]):
            planar_distance = np.linalg.norm(ground_points[:, :2] - position[None, :2], axis=1)
            ground_points = ground_points[planar_distance > radius]
    if len(ground_points) < 3:
        return TraversableRegion(ground_points, np.zeros((0, 3), np.float32), ground.confidence)
    if len(ground_points) > 3000:
        ground_points = ground_points[np.linspace(0, len(ground_points) - 1, 3000, dtype=np.int64)]
    try:
        hull = ConvexHull(ground_points[:, :2])
        polygon = ground_points[hull.vertices]
    except Exception:
        polygon = np.zeros((0, 3), dtype=np.float32)
    return TraversableRegion(ground_points.astype(np.float32), polygon.astype(np.float32), ground.confidence)
