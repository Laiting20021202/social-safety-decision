"""Small world-frame obstacle cache for the existing perception controller.

No simulator API or obstacle ground truth. Missing detections never carve free
space. Only subsequent measured rays passing behind an occupied voxel do so.
The uncertainty is an explicit engineering bound, not calibrated confidence.
"""
from __future__ import annotations

import numpy as np


class VisibilityMemory:
    def __init__(self, voxel_m=0.025, capacity=2048, confirmations=3):
        self.voxel_m = float(voxel_m)
        self.capacity = int(capacity)
        self.confirmations = int(confirmations)
        self.cells: dict[tuple[int, ...], tuple[np.ndarray, float, int]] = {}
        self.last_observed = -float("inf")
        self.last_scene = -float("inf")
        self.capacity_exceeded = False
        self.velocity = np.zeros(3)
        self.cell_velocity = {}

    def observe(self, points, stamp, velocity):
        if stamp <= self.last_observed or not len(points):
            return
        self.last_observed = float(stamp)
        self.velocity = np.asarray(velocity, dtype=float).copy()
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]
        keys, indices = np.unique(
            np.floor(points / self.voxel_m).astype(int), axis=0, return_index=True
        )
        for key, index in zip(keys, indices):
            self.cells[tuple(key)] = (points[index].copy(), float(stamp), 0)
            self.cell_velocity[tuple(key)] = self.velocity.copy()
        if len(self.cells) > self.capacity:
            # Bounded storage must not silently imply that discarded space is safe.
            self.capacity_exceeded = True
            self.cells = dict(sorted(
                self.cells.items(), key=lambda item: item[1][1], reverse=True
            )[:self.capacity])
            self.cell_velocity = {key: self.cell_velocity[key] for key in self.cells}

    def carve(self, optical_points, rotation, origin, stamp):
        """Carve with fresh optical rays (z forward), never occluded/absent rays.

        A ray bin uses the NEAREST measured return, avoiding false clearing
        through the arm. Three distinct frames must agree; old/repeated frames
        and frames preceding the obstacle measurement cannot erase memory.
        """
        if stamp <= self.last_scene:
            return 0
        self.last_scene = float(stamp)
        points = np.asarray(optical_points)
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)]
        if not len(points) or not self.cells:
            return 0
        step = 0.02
        bins, inverse = np.unique(
            np.floor(points[:, :2] / points[:, 2:3] / step).astype(int),
            axis=0, return_inverse=True,
        )
        depths = np.full(len(bins), np.inf)
        np.minimum.at(depths, inverse, points[:, 2])
        rays = {tuple(key): depth for key, depth in zip(bins, depths)}
        removed = 0
        for key, (world, seen, hits) in list(self.cells.items()):
            if stamp <= seen + 0.10:
                continue
            camera = (world - origin) @ rotation
            clear = False
            if camera[2] > 0.05:
                pixel = tuple(np.floor(camera[:2] / camera[2] / step).astype(int))
                depth = rays.get(pixel)
                clear = depth is not None and depth > camera[2] + 0.08
                if depth is not None and abs(depth - camera[2]) < 0.025:
                    # A fresh coincident surface confirms occupancy even when
                    # the semantic hand detector does not label this frame.
                    # Its age must not keep inflating a clearly visible table.
                    self.cells[key] = (world, float(stamp), 0)
                    self.cell_velocity[key] = np.zeros(3)
                    continue
            hits = hits + 1 if clear else 0
            if hits >= self.confirmations:
                del self.cells[key]
                self.cell_velocity.pop(key, None)
                removed += 1
            else:
                self.cells[key] = (world, seen, hits)
        return removed

    def snapshot(self, now):
        if not self.cells:
            return np.empty((0, 3)), np.empty(0), np.empty(0)
        points = np.asarray([cell[0] for cell in self.cells.values()])
        ages = np.maximum(0, now - np.asarray([cell[1] for cell in self.cells.values()]))
        # Keep the measured location in the collision envelope too: motion
        # may stop or reverse behind an occluder. Prediction is bounded in time.
        elapsed = np.minimum(np.maximum(ages - 0.25, 0), 0.5)
        velocities = np.asarray([self.cell_velocity[key] for key in self.cells])
        predicted = points + elapsed[:, None] * velocities
        moving = np.linalg.norm(velocities, axis=1) > .02
        # Confirmed stationary surfaces retain their pose-error allowance.
        # Only recently moving surfaces acquire an expanding hidden-motion
        # envelope; the absence of a view alone does not make a table move.
        uncertainty = np.where(moving,
            np.minimum(0.12, np.maximum(ages - 0.25, 0) * 0.025),
            np.minimum(.005, np.maximum(ages - 0.25, 0) * .005))
        return np.concatenate((points, predicted)), np.tile(ages, 2), np.tile(uncertainty, 2)


def memory_collision_boxes(points, uncertainty, padding=.015, resolution=.05):
    """Compact occupied cells; never fill the gap between separate obstacles.

    Uncertainty applies to each local cell, not to every observation using the
    oldest point's bound. Both measured and predicted samples are provided by
    snapshot(), so no second global velocity sweep is applied here.
    """
    points = np.asarray(points).reshape(-1, 3)
    uncertainty = np.broadcast_to(uncertainty, (len(points),))
    keys, inverse = np.unique(np.floor(points / resolution).astype(int), axis=0, return_inverse=True)
    boxes = []
    for index in range(len(keys)):
        mask = inverse == index
        chunk = points[mask]
        lower, upper = chunk.min(axis=0), chunk.max(axis=0)
        margin = padding + float(uncertainty[mask].max())
        boxes.append(((lower + upper) / 2, np.maximum(upper - lower, .01) + 2 * margin))
    return boxes
