from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np

from realtime_safety.config import ReconstructionConfig
from realtime_safety.types import PointCloudFrame


LOGGER = logging.getLogger(__name__)


class AprilTagScaleCalibrator:
    """Lock monocular point-cloud scale to a square AprilTag of known size.

    Detection happens in the original RGB image.  Its corner pixels are mapped
    into the dense reconstructed point map, where the four physical edges and
    two diagonals provide six independent scale estimates.  A rejected or
    temporarily hidden tag never changes the current metric scale.
    """

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        dictionary_id = getattr(cv2.aruco, config.apriltag_dictionary, None)
        if dictionary_id is None:
            raise ValueError(
                f"Unsupported AprilTag dictionary: {config.apriltag_dictionary}"
            )
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._scale: float | None = None
        self._age_frames = config.apriltag_hold_frames + 1
        self._last_tag_id: int | None = None
        self._last_observed_edge_m: float | None = None
        self._last_center_raw: np.ndarray | None = None
        self._last_corners_raw: np.ndarray | None = None
        window = max(config.apriltag_warmup_detections * 2, 7)
        self._scale_candidates: deque[float] = deque(maxlen=window)
        self._edge_candidates: deque[float] = deque(maxlen=window)

    @property
    def scale(self) -> float | None:
        return self._scale

    def reset(self) -> None:
        self._scale = None
        self._age_frames = self.config.apriltag_hold_frames + 1
        self._last_tag_id = None
        self._last_observed_edge_m = None
        self._last_center_raw = None
        self._last_corners_raw = None
        self._scale_candidates.clear()
        self._edge_candidates.clear()

    def calibrate(
        self,
        bgr: np.ndarray,
        cloud: PointCloudFrame,
    ) -> PointCloudFrame:
        """Apply one global scale correction in place and attach tag metadata."""

        should_detect = (
            cloud.frame_index % max(self.config.apriltag_detection_interval, 1)
            == 0
        )
        detected = False
        if should_detect:
            result = self._measure(bgr, cloud.pointmap)
            if result is not None:
                tag_id, candidate, edge, center, corners = result
                if self._accept_candidate(candidate):
                    self._scale_candidates.append(candidate)
                    self._edge_candidates.append(edge)
                    first_lock = self._scale is None
                    if self._scale is None:
                        if (
                            len(self._scale_candidates)
                            < self.config.apriltag_warmup_detections
                        ):
                            self._age_frames = 0
                            return cloud
                        self._scale = float(np.median(self._scale_candidates))
                    else:
                        robust_candidate = float(np.median(self._scale_candidates))
                        # A physical reference should correct slow model drift,
                        # not copy per-frame corner/depth noise into the world.
                        # Bound the accepted target to 1% before applying EMA.
                        robust_candidate = float(
                            np.clip(
                                robust_candidate,
                                self._scale * 0.99,
                                self._scale * 1.01,
                            )
                        )
                        alpha = float(self.config.apriltag_scale_ema_alpha)
                        self._scale = (
                            (1.0 - alpha) * self._scale
                            + alpha * robust_candidate
                        )
                    self._last_tag_id = tag_id
                    self._last_observed_edge_m = float(
                        np.median(self._edge_candidates)
                    )
                    self._last_center_raw = center
                    self._last_corners_raw = corners
                    self._age_frames = 0
                    detected = True
                    if first_lock:
                        LOGGER.info(
                            "AprilTag metric scale locked: id=%d physical_side=%.3fm "
                            "observed_side=%.4f model_units correction=%.5fx",
                            tag_id,
                            self.config.apriltag_size_m,
                            edge,
                            self._scale,
                        )
        if not detected:
            self._age_frames += 1

        if self._scale is None:
            cloud.apriltag_locked = False
            cloud.apriltag_age_frames = None
            return cloud

        scale = float(self._scale)
        cloud.points *= scale
        cloud.pointmap *= scale
        if cloud.tracking_points is not None:
            cloud.tracking_points *= scale
        if cloud.camera_transform is not None:
            cloud.camera_transform[:3, 3] *= scale
        cloud.metric_scale = (
            scale
            if cloud.metric_scale is None
            else float(cloud.metric_scale) * scale
        )
        cloud.apriltag_locked = self._age_frames <= self.config.apriltag_hold_frames
        cloud.apriltag_id = self._last_tag_id
        cloud.apriltag_size_m = float(self.config.apriltag_size_m)
        cloud.apriltag_observed_edge_m = self._last_observed_edge_m
        cloud.apriltag_scale_correction = scale
        cloud.apriltag_age_frames = self._age_frames
        cloud.apriltag_center_xyz = (
            None
            if self._last_center_raw is None
            else (self._last_center_raw * scale).astype(np.float32)
        )
        cloud.apriltag_corners_xyz = (
            None
            if self._last_corners_raw is None
            else (self._last_corners_raw * scale).astype(np.float32)
        )
        if "apriltag" not in cloud.source:
            cloud.source += "_apriltag_metric"
        return cloud

    def _accept_candidate(self, candidate: float) -> bool:
        if not np.isfinite(candidate) or not 0.05 <= candidate <= 20.0:
            return False
        if self._scale is None:
            return True
        # After lock, a one-frame jump is normally a corner on an occluder.
        ratio = candidate / self._scale
        return 0.65 <= ratio <= 1.35

    def _measure(
        self,
        bgr: np.ndarray,
        pointmap: np.ndarray,
    ) -> tuple[int, float, float, np.ndarray, np.ndarray] | None:
        image = np.asarray(bgr)
        dense = np.asarray(pointmap, dtype=np.float32)
        if image.ndim != 3 or dense.ndim != 3 or dense.shape[2] != 3:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or not len(corners_list):
            return None

        candidates: list[tuple[float, int, np.ndarray]] = []
        requested_id = self.config.apriltag_id
        for corners, marker_id in zip(corners_list, ids.reshape(-1)):
            marker_id = int(marker_id)
            if requested_id is not None and marker_id != requested_id:
                continue
            pixels = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            area = abs(float(cv2.contourArea(pixels)))
            candidates.append((area, marker_id, pixels))
        if not candidates:
            return None
        _, marker_id, pixels = max(candidates, key=lambda item: item[0])

        height, width = dense.shape[:2]
        image_height, image_width = image.shape[:2]
        mapped = pixels.copy()
        mapped[:, 0] *= width / max(float(image_width), 1.0)
        mapped[:, 1] *= height / max(float(image_height), 1.0)
        points = np.stack(
            [self._sample_point(dense, x, y) for x, y in mapped], axis=0
        )
        if not np.isfinite(points).all():
            return None

        edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
        diagonal_pairs = ((0, 2), (1, 3))
        edges = np.asarray(
            [np.linalg.norm(points[a] - points[b]) for a, b in edge_pairs],
            dtype=np.float64,
        )
        diagonals = np.asarray(
            [np.linalg.norm(points[a] - points[b]) for a, b in diagonal_pairs],
            dtype=np.float64,
        )
        if np.any(edges <= 1e-4) or np.any(diagonals <= 1e-4):
            return None
        size = float(self.config.apriltag_size_m)
        ratios = np.concatenate(
            (size / edges, (np.sqrt(2.0) * size) / diagonals)
        )
        scale = float(np.median(ratios))
        spread = float(np.max(np.abs(ratios / scale - 1.0)))
        if spread > self.config.apriltag_max_ratio_spread:
            LOGGER.debug(
                "Rejected AprilTag %d scale: ratio spread %.1f%%",
                marker_id,
                spread * 100.0,
            )
            return None
        observed_edge = float(np.median(edges))
        return marker_id, scale, observed_edge, np.mean(points, axis=0), points

    @staticmethod
    def _sample_point(pointmap: np.ndarray, x: float, y: float) -> np.ndarray:
        height, width = pointmap.shape[:2]
        cx = int(np.clip(round(float(x)), 0, width - 1))
        cy = int(np.clip(round(float(y)), 0, height - 1))
        radius = 1
        patch = pointmap[
            max(0, cy - radius) : min(height, cy + radius + 1),
            max(0, cx - radius) : min(width, cx + radius + 1),
        ].reshape(-1, 3)
        valid = np.isfinite(patch).all(axis=1) & (patch[:, 1] > 0.01)
        if not np.any(valid):
            return np.full(3, np.nan, dtype=np.float32)
        return np.median(patch[valid], axis=0).astype(np.float32)


__all__ = ["AprilTagScaleCalibrator"]
