from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from social_bev.types import Detection, ObstacleRegion


class UnknownObstacleExtractor:
    """Extract low-confidence RGB-estimated obstacle regions not explained by known detections."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.ground_roi_start_ratio = float(config.get("ground_roi_start_ratio", 0.38))
        self.corridor_width_ratio = float(config.get("corridor_width_ratio", 0.42))
        self.minimum_area = float(config.get("minimum_area", 300))
        self.maximum_area_ratio = float(config.get("maximum_area_ratio", 0.30))
        self.morphology_kernel = int(config.get("morphology_kernel", 7))
        self.confidence = float(config.get("confidence", 0.35))

    def extract(
        self,
        walkable_mask: np.ndarray,
        detections: list[Detection],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[ObstacleRegion]:
        if not self.enabled:
            return []
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if h <= 0 or w <= 0:
            return []
        walkable = cv2.resize(
            walkable_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        roi = np.zeros((h, w), dtype=np.uint8)
        y0 = int(np.clip(self.ground_roi_start_ratio, 0.0, 0.95) * h)
        roi[y0:h, :] = 1

        corridor = np.zeros((h, w), dtype=np.uint8)
        half_width = int(max(8, self.corridor_width_ratio * w * 0.5))
        center = w // 2
        x0 = max(0, center - half_width)
        x1 = min(w, center + half_width)
        corridor[y0:h, x0:x1] = 1

        candidate = ((~walkable).astype(np.uint8) & roi & corridor)
        candidate = self._remove_known_detections(candidate, detections)
        if self.morphology_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morphology_kernel, self.morphology_kernel),
            )
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[ObstacleRegion] = []
        max_area = self.maximum_area_ratio * h * w
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.minimum_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if y + bh < y0:
                continue
            ground_points = self._lowest_contour_points(contour)
            regions.append(
                ObstacleRegion(
                    bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                    contour=contour,
                    confidence=self.confidence,
                    label="unknown_obstacle",
                    category="unknown_obstacle",
                    ground_points=ground_points,
                    note="RGB estimate",
                )
            )
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _remove_known_detections(self, candidate: np.ndarray, detections: list[Detection]) -> np.ndarray:
        mask = candidate.copy()
        h, w = mask.shape[:2]
        for detection in detections:
            if detection.category not in {"person", "known_obstacle"}:
                continue
            x1, y1, x2, y2 = detection.bbox
            pad_x = 0.06 * max(1.0, x2 - x1)
            pad_y = 0.08 * max(1.0, y2 - y1)
            xa = int(np.clip(x1 - pad_x, 0, w - 1))
            ya = int(np.clip(y1 - pad_y, 0, h - 1))
            xb = int(np.clip(x2 + pad_x, 0, w))
            yb = int(np.clip(y2 + pad_y, 0, h))
            mask[ya:yb, xa:xb] = 0
        return mask

    def _lowest_contour_points(self, contour: np.ndarray) -> list[tuple[float, float]]:
        pts = contour.reshape(-1, 2)
        if len(pts) == 0:
            return []
        max_y = int(np.max(pts[:, 1]))
        bottom = pts[np.abs(pts[:, 1] - max_y) <= 3]
        if len(bottom) == 0:
            bottom = pts
        left = bottom[int(np.argmin(bottom[:, 0]))]
        center = bottom[int(np.argmin(np.abs(bottom[:, 0] - np.median(bottom[:, 0]))))]
        right = bottom[int(np.argmax(bottom[:, 0]))]
        return [(float(left[0]), float(left[1])), (float(center[0]), float(center[1])), (float(right[0]), float(right[1]))]

