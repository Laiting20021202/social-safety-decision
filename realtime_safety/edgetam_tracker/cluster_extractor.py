from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from realtime_safety.edgetam_tracker.models import (
    AABB,
    OBB,
    CloudFrame,
    Cluster3D,
    PointCloudQuality,
)


@dataclass(slots=True)
class ClusterExtractorConfig:
    method: str = "dbscan"
    tolerance: float = 0.08
    min_points: int = 20
    max_points: int = 100_000
    min_dimension: float | np.ndarray = 0.0
    max_dimension: float | np.ndarray = np.inf
    dbscan_min_samples: int = 5
    depth_axis: int = 1
    sparse_point_threshold: int | None = None
    minimum_density: float = 0.0
    emergency_distance: float = 0.0
    emergency_min_points: int = 3

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method not in {"dbscan", "euclidean"}:
            raise ValueError("clustering method must be 'dbscan' or 'euclidean'")
        if self.tolerance <= 0:
            raise ValueError("cluster tolerance must be positive")
        if self.min_points < 1:
            raise ValueError("min_points must be positive")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be at least min_points")
        if self.dbscan_min_samples < 1:
            raise ValueError("dbscan_min_samples must be positive")
        if self.depth_axis not in {0, 1, 2}:
            raise ValueError("depth_axis must be 0, 1, or 2")
        if self.sparse_point_threshold is not None and self.sparse_point_threshold < 1:
            raise ValueError("sparse_point_threshold must be positive")
        if self.minimum_density < 0:
            raise ValueError("minimum_density cannot be negative")
        if self.emergency_distance < 0:
            raise ValueError("emergency_distance cannot be negative")
        if self.emergency_min_points < 1:
            raise ValueError("emergency_min_points must be positive")
        minimum = np.asarray(self.min_dimension, dtype=np.float32)
        maximum = np.asarray(self.max_dimension, dtype=np.float32)
        if np.any(minimum < 0) or np.any(maximum < minimum):
            raise ValueError("cluster dimensions must satisfy 0 <= min <= max")


ClusteringConfig = ClusterExtractorConfig


class ClusterExtractor:
    def __init__(
        self,
        config: ClusterExtractorConfig | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = ClusterExtractorConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config

    def extract(
        self,
        cloud: CloudFrame,
        robot_origin: np.ndarray | None = None,
    ) -> list[Cluster3D]:
        points = np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return []
        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            cloud = cloud.select(finite)
            points = cloud.points
        tree = cKDTree(points)
        if self.config.method == "dbscan":
            labels = self._dbscan_labels(points, tree)
        else:
            labels = self._euclidean_labels(points, tree)

        origin = (
            np.zeros(3, dtype=np.float32)
            if robot_origin is None
            else np.asarray(robot_origin, dtype=np.float32).reshape(3)
        )
        clusters: list[Cluster3D] = []
        accepted = np.zeros(len(points), dtype=bool)
        unique_labels = sorted(int(label) for label in np.unique(labels) if label >= 0)
        for label in unique_labels:
            indices = np.flatnonzero(labels == label)
            if not self.config.min_points <= len(indices) <= self.config.max_points:
                continue
            cluster_points = points[indices]
            aabb = AABB(
                minimum=np.min(cluster_points, axis=0),
                maximum=np.max(cluster_points, axis=0),
            )
            if not self._dimensions_are_valid(aabb.size):
                continue
            obb = self._pca_obb(cluster_points)
            delta = cluster_points - origin
            nearest_index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
            nearest_point = cluster_points[nearest_index]
            nearest_distance = float(np.linalg.norm(delta[nearest_index]))
            volume_floor = max(float(self.config.tolerance) ** 3, 1e-9)
            volume = max(float(np.prod(np.maximum(obb.size, 0.0))), volume_floor)
            density = float(len(cluster_points) / volume)
            sparse_threshold = (
                self.config.sparse_point_threshold
                if self.config.sparse_point_threshold is not None
                else self.config.min_points
            )
            quality = (
                PointCloudQuality.GOOD
                if len(cluster_points) >= sparse_threshold
                and density >= self.config.minimum_density
                else PointCloudQuality.SPARSE
            )
            point_score = min(len(cluster_points) / max(float(sparse_threshold), 1.0), 1.0)
            density_score = (
                1.0
                if self.config.minimum_density <= 0
                else min(density / self.config.minimum_density, 1.0)
            )
            cluster_pixels = (
                None
                if cloud.pixels_uv is None
                else cloud.pixels_uv[indices]
            )
            clusters.append(
                Cluster3D(
                    cluster_id=len(clusters),
                    points=cluster_points,
                    colors=None if cloud.colors is None else cloud.colors[indices],
                    source_indices=(
                        None
                        if cloud.source_indices is None
                        else cloud.source_indices[indices]
                    ),
                    pixels_uv=cluster_pixels,
                    centroid=np.mean(cluster_points, axis=0),
                    median_center=np.median(cluster_points, axis=0),
                    aabb=aabb,
                    obb=obb,
                    nearest_point=nearest_point,
                    nearest_distance=nearest_distance,
                    point_count=len(cluster_points),
                    density=density,
                    depth_variance=float(
                        np.var(cluster_points[:, self.config.depth_axis])
                    ),
                    missing_depth_ratio=self._missing_depth_ratio(
                        cluster_pixels,
                        cloud.image_shape,
                    ),
                    quality=quality,
                    quality_score=float(0.7 * point_score + 0.3 * density_score),
                )
            )
            accepted[indices] = True

        # The regular minimum-point/size filters are noise controls, not a
        # license to erase a handful of returns immediately next to the robot.
        # Conservatively group residual near-origin returns with a lower,
        # explicit floor. They stay INVALID/low-confidence so downstream code
        # increases uncertainty and never lets a mask narrow them.
        if self.config.emergency_distance > 0.0:
            residual_indices = np.flatnonzero(~accepted)
            for component in self._residual_components(
                points,
                residual_indices,
                tree,
            ):
                if len(component) < self.config.emergency_min_points:
                    continue
                component_points = points[component]
                delta = component_points - origin
                squared = np.einsum("ij,ij->i", delta, delta)
                nearest_index = int(np.argmin(squared))
                nearest_distance = float(np.sqrt(squared[nearest_index]))
                if nearest_distance > self.config.emergency_distance:
                    continue
                aabb = AABB(
                    minimum=np.min(component_points, axis=0),
                    maximum=np.max(component_points, axis=0),
                )
                volume = max(
                    float(np.prod(np.maximum(aabb.size, 0.0))),
                    max(float(self.config.tolerance) ** 3, 1e-9),
                )
                component_pixels = (
                    None
                    if cloud.pixels_uv is None
                    else cloud.pixels_uv[component]
                )
                clusters.append(
                    Cluster3D(
                        cluster_id=len(clusters),
                        points=component_points,
                        colors=(
                            None
                            if cloud.colors is None
                            else cloud.colors[component]
                        ),
                        source_indices=(
                            None
                            if cloud.source_indices is None
                            else cloud.source_indices[component]
                        ),
                        pixels_uv=component_pixels,
                        centroid=np.mean(component_points, axis=0),
                        median_center=np.median(component_points, axis=0),
                        aabb=aabb,
                        obb=self._pca_obb(component_points),
                        nearest_point=component_points[nearest_index],
                        nearest_distance=nearest_distance,
                        point_count=len(component_points),
                        density=float(len(component_points) / volume),
                        depth_variance=float(
                            np.var(
                                component_points[
                                    :, self.config.depth_axis
                                ]
                            )
                        ),
                        missing_depth_ratio=self._missing_depth_ratio(
                            component_pixels,
                            cloud.image_shape,
                        ),
                        quality=PointCloudQuality.INVALID,
                        quality_score=0.05,
                    )
                )
        return clusters

    process = extract

    @staticmethod
    def _missing_depth_ratio(
        pixels_uv: np.ndarray | None,
        image_shape: tuple[int, int] | None,
    ) -> float:
        """Estimate enclosed holes in the cluster's observed 2D footprint.

        Empty pixels connected to the footprint boundary remain unknown.  This
        prevents voxel downsampling gaps from being mislabeled as missing depth.
        """

        if pixels_uv is None or image_shape is None:
            return 0.0
        try:
            height, width = (int(image_shape[0]), int(image_shape[1]))
            pixels = np.asarray(pixels_uv, dtype=np.int64).reshape(-1, 2)
        except (IndexError, TypeError, ValueError):
            return 0.0
        if height <= 0 or width <= 0 or len(pixels) == 0:
            return 0.0
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        if not valid.any():
            return 0.0
        occupied = np.unique(pixels[valid], axis=0)
        minimum = np.min(occupied, axis=0)
        maximum = np.max(occupied, axis=0)
        occupancy = np.zeros(
            (
                int(maximum[1] - minimum[1] + 1),
                int(maximum[0] - minimum[0] + 1),
            ),
            dtype=bool,
        )
        local_pixels = occupied - minimum
        occupancy[local_pixels[:, 1], local_pixels[:, 0]] = True
        filled_footprint = np.asarray(
            binary_fill_holes(occupancy),
            dtype=bool,
        )
        footprint_area = int(np.count_nonzero(filled_footprint))
        if footprint_area <= 0:
            return 0.0
        hole_count = footprint_area - len(occupied)
        return float(
            np.clip(hole_count / float(footprint_area), 0.0, 1.0)
        )

    def _euclidean_labels(self, points: np.ndarray, tree: cKDTree) -> np.ndarray:
        labels = np.full(len(points), -1, dtype=np.int32)
        visited = np.zeros(len(points), dtype=bool)
        next_label = 0
        for start in range(len(points)):
            if visited[start]:
                continue
            visited[start] = True
            component: list[int] = []
            frontier: deque[int] = deque([start])
            while frontier:
                index = frontier.popleft()
                component.append(index)
                neighbors = tree.query_ball_point(points[index], self.config.tolerance)
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        frontier.append(neighbor)
            if len(component) >= self.config.min_points:
                labels[np.asarray(component, dtype=np.int64)] = next_label
                next_label += 1
        return labels

    def _residual_components(
        self,
        points: np.ndarray,
        residual_indices: np.ndarray,
        tree: cKDTree,
    ) -> list[np.ndarray]:
        residual = set(int(index) for index in residual_indices)
        components: list[np.ndarray] = []
        while residual:
            start = min(residual)
            residual.remove(start)
            component = [start]
            frontier: deque[int] = deque([start])
            while frontier:
                index = frontier.popleft()
                for neighbor in tree.query_ball_point(
                    points[index],
                    self.config.tolerance,
                ):
                    neighbor = int(neighbor)
                    if neighbor in residual:
                        residual.remove(neighbor)
                        component.append(neighbor)
                        frontier.append(neighbor)
            components.append(np.asarray(component, dtype=np.int64))
        return components

    def _dbscan_labels(self, points: np.ndarray, tree: cKDTree) -> np.ndarray:
        unvisited = -2
        noise = -1
        labels = np.full(len(points), unvisited, dtype=np.int32)
        next_label = 0
        minimum_samples = int(self.config.dbscan_min_samples)
        for start in range(len(points)):
            if labels[start] != unvisited:
                continue
            neighbors = tree.query_ball_point(points[start], self.config.tolerance)
            if len(neighbors) < minimum_samples:
                labels[start] = noise
                continue

            labels[start] = next_label
            frontier: deque[int] = deque(int(index) for index in neighbors if index != start)
            queued = np.zeros(len(points), dtype=bool)
            if frontier:
                queued[np.fromiter(frontier, dtype=np.int64)] = True
            while frontier:
                index = frontier.popleft()
                if labels[index] == noise:
                    labels[index] = next_label
                if labels[index] != unvisited:
                    continue
                labels[index] = next_label
                local_neighbors = tree.query_ball_point(
                    points[index],
                    self.config.tolerance,
                )
                if len(local_neighbors) < minimum_samples:
                    continue
                for neighbor in local_neighbors:
                    neighbor = int(neighbor)
                    if labels[neighbor] in {unvisited, noise} and not queued[neighbor]:
                        frontier.append(neighbor)
                        queued[neighbor] = True
            next_label += 1
        return labels

    def _dimensions_are_valid(self, size: np.ndarray) -> bool:
        minimum = np.asarray(self.config.min_dimension, dtype=np.float32)
        maximum = np.asarray(self.config.max_dimension, dtype=np.float32)
        if minimum.ndim == 0:
            if float(np.max(size)) < float(minimum):
                return False
        elif np.any(size < np.broadcast_to(minimum, (3,))):
            return False
        if maximum.ndim == 0:
            if float(np.max(size)) > float(maximum):
                return False
        elif np.any(size > np.broadcast_to(maximum, (3,))):
            return False
        return True

    @staticmethod
    def _pca_obb(points: np.ndarray) -> OBB:
        center = np.mean(points, axis=0, dtype=np.float64)
        centered = np.asarray(points, dtype=np.float64) - center
        if len(points) <= 1 or np.allclose(centered, 0.0):
            rotation = np.eye(3, dtype=np.float64)
        else:
            covariance = centered.T @ centered / max(len(points) - 1, 1)
            _, eigenvectors = np.linalg.eigh(covariance)
            rotation = eigenvectors[:, ::-1]
            if np.linalg.det(rotation) < 0:
                rotation[:, -1] *= -1.0
        local = centered @ rotation
        local_minimum = np.min(local, axis=0)
        local_maximum = np.max(local, axis=0)
        local_center = (local_minimum + local_maximum) * 0.5
        world_center = center + rotation @ local_center
        return OBB(
            center=world_center,
            size=np.maximum(local_maximum - local_minimum, 0.0),
            rotation=rotation,
        )


__all__ = [
    "ClusterExtractor",
    "ClusterExtractorConfig",
    "ClusteringConfig",
]
