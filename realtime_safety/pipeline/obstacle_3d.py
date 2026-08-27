from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from realtime_safety.pipeline.pointcloud import voxel_downsample
from realtime_safety.types import BBox3D, Detection2D, ObstacleObservation3D, PointCloudFrame


@dataclass(slots=True)
class DetectionDepthDiagnostics:
    track_id: int | None
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_pixels: int
    sampled_mask_pixels: int
    valid_depth_pixels: int
    depth_min_m: float | None
    depth_median_m: float | None
    depth_max_m: float | None
    filtered_points: int
    output_points: int
    reason: str


class ObstacleExtractor3D:
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        max_depth: float = 20.0,
        voxel_size: float = 0.05,
        minimum_points: int = 12,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.minimum_points = minimum_points
        self.last_diagnostics: list[DetectionDepthDiagnostics] = []

    def extract(
        self, detections: list[Detection2D], cloud: PointCloudFrame
    ) -> tuple[list[ObstacleObservation3D], np.ndarray]:
        height, width = cloud.pointmap.shape[:2]
        dense_confidence = (
            cloud.dense_confidence
            if cloud.dense_confidence is not None and cloud.dense_confidence.shape == (height, width)
            else np.ones((height, width), dtype=np.float32)
        )
        assigned = np.zeros((height, width), dtype=bool)
        observations: list[ObstacleObservation3D] = []
        diagnostics: list[DetectionDepthDiagnostics] = []
        valid_geometry = (
            np.isfinite(cloud.pointmap).all(axis=-1)
            & (dense_confidence >= self.confidence_threshold)
            & (cloud.pointmap[..., 1] > 0.05)
            & (cloud.pointmap[..., 1] < self.max_depth)
        )
        for detection in detections:
            mask = self._mask_for_detection(detection, width, height)
            assignment_mask = mask
            mask_pixels = int(mask.sum())
            if detection.class_name in {"person", "hand", "human_hand"} and int(mask.sum()) >= 80:
                # Trim mixed foreground/background boundary pixels before
                # reading the St4RTrack pointmap.
                mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            sampled_mask_pixels = int(mask.sum())
            valid = mask & valid_geometry
            points = cloud.pointmap[valid]
            forward_depth = points[:, 1] if len(points) else np.empty((0,), dtype=np.float32)
            depth_min = float(np.min(forward_depth)) if len(forward_depth) else None
            depth_median = float(np.median(forward_depth)) if len(forward_depth) else None
            depth_max = float(np.max(forward_depth)) if len(forward_depth) else None
            if detection.track_id is None:
                diagnostics.append(
                    self._diagnostic(
                        detection,
                        mask_pixels,
                        sampled_mask_pixels,
                        len(points),
                        depth_min,
                        depth_median,
                        depth_max,
                        filtered_points=0,
                        output_points=0,
                        reason="missing_track_id",
                    )
                )
                continue
            if detection.class_name == "person":
                points = self._robust_filter_person(points)
            elif detection.class_name in {"hand", "human_hand"}:
                points = self._robust_filter_hand(points)
            else:
                points = self._robust_filter(points)
            filtered_points = len(points)
            if len(points) < self.minimum_points:
                diagnostics.append(
                    self._diagnostic(
                        detection,
                        mask_pixels,
                        sampled_mask_pixels,
                        int(valid.sum()),
                        depth_min,
                        depth_median,
                        depth_max,
                        filtered_points,
                        output_points=0,
                        reason="insufficient_points_after_depth_filter",
                    )
                )
                continue
            # Keep the full segmentation footprint reserved even though person
            # geometry is estimated from a boundary-trimmed mask.
            assigned |= assignment_mask & valid_geometry
            confidence = np.full(len(points), detection.confidence, dtype=np.float32)
            points, _, _ = voxel_downsample(
                points,
                np.zeros((len(points), 3), dtype=np.uint8),
                confidence,
                self.voxel_size,
                max_points=3000,
            )
            observations.append(self._observation(detection.track_id, detection.class_name, detection.confidence, points, cloud.timestamp))
            diagnostics.append(
                self._diagnostic(
                    detection,
                    mask_pixels,
                    sampled_mask_pixels,
                    int(valid.sum()),
                    depth_min,
                    depth_median,
                    depth_max,
                    filtered_points,
                    output_points=len(points),
                    reason="ok",
                )
            )
        self.last_diagnostics = diagnostics
        return observations, assigned

    @staticmethod
    def _diagnostic(
        detection: Detection2D,
        mask_pixels: int,
        sampled_mask_pixels: int,
        valid_depth_pixels: int,
        depth_min_m: float | None,
        depth_median_m: float | None,
        depth_max_m: float | None,
        filtered_points: int,
        output_points: int,
        reason: str,
    ) -> DetectionDepthDiagnostics:
        return DetectionDepthDiagnostics(
            track_id=detection.track_id,
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
            mask_pixels=int(mask_pixels),
            sampled_mask_pixels=int(sampled_mask_pixels),
            valid_depth_pixels=int(valid_depth_pixels),
            depth_min_m=depth_min_m,
            depth_median_m=depth_median_m,
            depth_max_m=depth_max_m,
            filtered_points=int(filtered_points),
            output_points=int(output_points),
            reason=reason,
        )

    def find_unknown(
        self,
        cloud: PointCloudFrame,
        assigned: np.ndarray,
        start_track_id: int = -1,
        eps: float = 0.3,
        minimum_points: int = 30,
    ) -> list[ObstacleObservation3D]:
        """Radius-connected clustering for unassigned, non-ground-like near points."""
        height, width = cloud.pointmap.shape[:2]
        if assigned.shape != (height, width):
            raise ValueError("assigned mask shape does not match pointmap")
        points = cloud.pointmap.reshape(-1, 3)
        valid = ~assigned.reshape(-1) & np.isfinite(points).all(axis=1)
        valid &= (points[:, 1] > 0.2) & (points[:, 1] < min(self.max_depth, 8.0))
        # Exclude most floor/ceiling and keep the front safety volume.
        valid &= (points[:, 2] > -1.2) & (points[:, 2] < 2.5) & (np.abs(points[:, 0]) < 5.0)
        points = points[valid]
        if len(points) > 4000:
            points = points[np.linspace(0, len(points) - 1, 4000, dtype=np.int64)]
        if len(points) < minimum_points:
            return []
        graph = cKDTree(points).sparse_distance_matrix(cKDTree(points), eps, output_type="coo_matrix")
        count, labels = connected_components(graph, directed=False)
        observations: list[ObstacleObservation3D] = []
        next_id = start_track_id
        for label in range(count):
            cluster = self._robust_filter(points[labels == label])
            if len(cluster) < minimum_points:
                continue
            observations.append(self._observation(next_id, "unknown_obstacle", 0.35, cluster, cloud.timestamp))
            next_id -= 1
        return observations

    @staticmethod
    def _mask_for_detection(detection: Detection2D, width: int, height: int) -> np.ndarray:
        if detection.mask is not None:
            return cv2.resize(
                detection.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        x1, y1, x2, y2 = detection.bbox_xyxy
        source_width, source_height = detection.image_size or (max(float(x2), width), max(float(y2), height))
        sx, sy = width / source_width, height / source_height
        result = np.zeros((height, width), dtype=bool)
        result[max(0, int(y1 * sy)) : min(height, int(np.ceil(y2 * sy))), max(0, int(x1 * sx)) : min(width, int(np.ceil(x2 * sx)))] = True
        return result

    @staticmethod
    def _robust_filter(points: np.ndarray) -> np.ndarray:
        if len(points) < 8:
            return points
        median = np.median(points, axis=0)
        distance = np.linalg.norm(points - median, axis=1)
        mad = np.median(np.abs(distance - np.median(distance)))
        threshold = np.median(distance) + max(3.5 * mad, 0.1)
        return points[distance <= threshold]

    @classmethod
    def _robust_filter_person(cls, points: np.ndarray) -> np.ndarray:
        """Keep the compact foreground depth layer inside a person mask."""
        points = cls._robust_filter(points)
        if len(points) < 12:
            return points
        forward = points[:, 1]
        median = float(np.median(forward))
        mad = float(np.median(np.abs(forward - median)))
        depth_gate = max(3.0 * 1.4826 * mad, 0.035 * max(abs(median), 1.0), 0.025)
        points = points[np.abs(forward - median) <= depth_gate]
        if len(points) < 12:
            return points
        lower, upper = np.percentile(points, (5.0, 95.0), axis=0)
        span = np.maximum(upper - lower, 0.02)
        inside = np.all((points >= lower - 0.25 * span) & (points <= upper + 0.25 * span), axis=1)
        return points[inside]

    def _robust_filter_hand(self, points: np.ndarray) -> np.ndarray:
        """Keep the nearest dense RGB-D layer inside a hand mask.

        A hand is above the table and therefore closer to the camera.  Using
        the median depth of the complete segmentation mask selects the table
        whenever mask edges/background outnumber hand pixels.  Find the first
        sufficiently populated depth layer instead, then trim its spatial
        outliers.  Isolated near-depth noise cannot satisfy the population
        gate.
        """
        if len(points) < self.minimum_points:
            return points

        forward = points[:, 1]
        order = np.argsort(forward)
        sorted_forward = forward[order]
        minimum_layer_points = max(
            self.minimum_points,
            int(np.ceil(0.06 * len(sorted_forward))),
        )
        window_m = 0.10
        layer_start: float | None = None
        layer_end: float | None = None
        for start_index, start_depth in enumerate(sorted_forward):
            end_index = int(
                np.searchsorted(sorted_forward, start_depth + window_m, side="right")
            )
            if end_index - start_index >= minimum_layer_points:
                layer_start = float(start_depth)
                # Include the full hand surface while retaining a clear gap
                # to the table/background behind it.
                layer_end = float(start_depth + 0.14)
                break

        if layer_start is None or layer_end is None:
            return self._robust_filter(points)

        selected = points[(forward >= layer_start - 0.01) & (forward <= layer_end)]
        if len(selected) < self.minimum_points:
            return self._robust_filter(points)
        selected = self._robust_filter(selected)
        if len(selected) < self.minimum_points:
            return selected
        lower, upper = np.percentile(selected, (2.0, 98.0), axis=0)
        span = np.maximum(upper - lower, 0.015)
        inside = np.all(
            (selected >= lower - 0.15 * span)
            & (selected <= upper + 0.15 * span),
            axis=1,
        )
        return selected[inside]

    @staticmethod
    def _observation(track_id: int, class_name: str, confidence: float, points: np.ndarray, timestamp: float) -> ObstacleObservation3D:
        percentiles = [5.0, 95.0] if class_name == "person" else [2.0, 98.0]
        minimum, maximum = np.percentile(points, percentiles, axis=0).astype(np.float32)
        bbox = BBox3D(minimum=minimum, maximum=maximum)
        center = np.median(points, axis=0).astype(np.float32)
        radius = max(float(np.linalg.norm((maximum[:2] - minimum[:2]) * 0.5)), 0.05)
        return ObstacleObservation3D(
            track_id=track_id,
            class_name=class_name,
            confidence=float(confidence),
            position_xyz=center,
            bbox3d=bbox,
            radius=radius,
            point_count=len(points),
            timestamp=float(timestamp),
            points=points.astype(np.float32),
        )
