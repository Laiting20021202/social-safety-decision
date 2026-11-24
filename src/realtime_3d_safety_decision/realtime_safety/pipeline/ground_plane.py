from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GroundPlane:
    coefficients: np.ndarray
    inlier_mask: np.ndarray
    confidence: float

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, b, c, d = self.coefficients
        return -(a * x + b * y + d) / max(abs(float(c)), 1e-6) * np.sign(c or 1.0)


class GroundPlaneEstimator:
    def __init__(self, distance_threshold: float = 0.08, iterations: int = 80, camera_height: float | None = None) -> None:
        self.distance_threshold = distance_threshold
        self.iterations = iterations
        self.camera_height = camera_height
        self._smoothed: np.ndarray | None = None
        self._rng = np.random.default_rng(31)

    def estimate(self, points: np.ndarray) -> GroundPlane | None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        points = points[(points[:, 1] > 0.2) & (points[:, 1] < 15.0)]
        if len(points) > 5000:
            points = points[self._rng.choice(len(points), 5000, replace=False)]
        if len(points) < 50:
            return None
        best_coefficients: np.ndarray | None = None
        best_inliers = np.zeros(len(points), dtype=bool)
        for _ in range(self.iterations):
            sample = points[self._rng.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-5:
                continue
            normal /= norm
            if abs(float(normal[2])) < 0.75:
                continue
            if normal[2] < 0:
                normal = -normal
            d = -float(normal @ sample[0])
            distances = np.abs(points @ normal + d)
            inliers = distances <= self.distance_threshold
            # Prefer lower horizontal surfaces and optionally camera-height agreement.
            score = int(inliers.sum())
            if self.camera_height is not None:
                height = abs(d / max(float(normal[2]), 1e-6))
                score -= int(abs(height - self.camera_height) * 30)
            if score > int(best_inliers.sum()):
                best_coefficients = np.r_[normal, d]
                best_inliers = inliers
        if best_coefficients is None or best_inliers.sum() < 30:
            return None
        inlier_points = points[best_inliers]
        centered = inlier_points - inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        coefficients = np.r_[normal, -normal @ inlier_points.mean(axis=0)]
        coefficients /= max(np.linalg.norm(coefficients[:3]), 1e-6)
        if self._smoothed is not None and np.dot(self._smoothed[:3], coefficients[:3]) > 0.9:
            coefficients = 0.8 * self._smoothed + 0.2 * coefficients
            coefficients /= max(np.linalg.norm(coefficients[:3]), 1e-6)
        self._smoothed = coefficients
        distances = np.abs(points @ coefficients[:3] + coefficients[3])
        inliers = distances <= self.distance_threshold
        return GroundPlane(coefficients.astype(np.float32), inliers, float(inliers.mean()))

    def reset(self) -> None:
        self._smoothed = None
