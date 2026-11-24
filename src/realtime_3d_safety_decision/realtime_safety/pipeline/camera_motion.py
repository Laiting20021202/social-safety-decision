from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CameraMotion:
    affine_2d: np.ndarray
    confidence: float
    tracked_points: int


class CameraMotionEstimator:
    def __init__(self, max_corners: int = 300) -> None:
        self.max_corners = max_corners
        self._previous_gray: np.ndarray | None = None

    def update(self, bgr: np.ndarray, exclusion_mask: np.ndarray | None = None) -> CameraMotion:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            return CameraMotion(identity, 0.0, 0)
        previous_gray = self._previous_gray
        feature_mask = None if exclusion_mask is None else (~exclusion_mask.astype(bool)).astype(np.uint8) * 255
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray, self.max_corners, 0.01, 8, mask=feature_mask, blockSize=7
        )
        self._previous_gray = gray
        if previous_points is None or len(previous_points) < 8:
            return CameraMotion(identity, 0.0, 0)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, previous_points, None
        )
        if current_points is None or status is None:
            return CameraMotion(identity, 0.0, 0)
        good_old = previous_points[status.ravel() == 1].reshape(-1, 2)
        good_new = current_points[status.ravel() == 1].reshape(-1, 2)
        if len(good_old) < 8:
            return CameraMotion(identity, 0.0, len(good_old))
        affine, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if affine is None or inliers is None:
            return CameraMotion(identity, 0.0, len(good_old))
        confidence = float(inliers.mean())
        return CameraMotion(affine.astype(np.float32), confidence, len(good_old))

    def reset(self) -> None:
        self._previous_gray = None
