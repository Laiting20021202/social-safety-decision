from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from realtime_safety.config import TrackingConfig
from realtime_safety.types import Detection2D


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-6)


@dataclass(slots=True)
class _Track2D:
    track_id: int
    bbox: np.ndarray
    centroid: np.ndarray
    velocity: np.ndarray
    class_name: str
    timestamp: float
    image_size: tuple[int, int] | None = None
    missing: int = 0
    hits: int = 1

    def predicted_bbox(self, timestamp: float) -> np.ndarray:
        dt = float(np.clip(timestamp - self.timestamp, 0.0, 1.0))
        offset = np.tile(self.velocity * dt, 2)
        return self.bbox + offset


class StableTracker2D:
    """Timestamp-aware IoU/Hungarian tracker for segmented instances."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._tracks: dict[int, _Track2D] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _cost(self, track: _Track2D, detection: Detection2D) -> float:
        if track.class_name != detection.class_name:
            return 1e3
        predicted = track.predicted_bbox(detection.timestamp)
        iou = bbox_iou(predicted, detection.bbox_xyxy)
        scale = max(np.linalg.norm(predicted[2:] - predicted[:2]), 1.0)
        distance = np.linalg.norm((predicted[:2] + predicted[2:]) * 0.5 - detection.centroid_xy) / scale
        return (1.0 - iou) + 0.25 * float(distance)

    def update(self, detections: list[Detection2D], timestamp: float) -> list[Detection2D]:
        track_ids = list(self._tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if track_ids and detections:
            costs = np.array(
                [[self._cost(self._tracks[track_id], det) for det in detections] for track_id in track_ids],
                dtype=np.float32,
            )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] > 1.0 - self.config.iou_threshold + 0.4:
                    continue
                track_id = track_ids[row]
                track = self._tracks[track_id]
                detection = detections[column]
                dt = max(detection.timestamp - track.timestamp, 1e-3)
                measured_velocity = (detection.centroid_xy - track.centroid) / dt
                track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
                track.bbox = detection.bbox_xyxy.copy()
                track.centroid = detection.centroid_xy.copy()
                track.timestamp = detection.timestamp
                track.missing = 0
                track.hits += 1
                detection.track_id = track_id
                detection.track_hits = track.hits
                detection.velocity_xy = track.velocity.astype(np.float32)
                matched_tracks.add(track_id)
                matched_detections.add(int(column))
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track2D(
                track_id=track_id,
                bbox=detection.bbox_xyxy.copy(),
                centroid=detection.centroid_xy.copy(),
                velocity=np.zeros(2, dtype=np.float32),
                class_name=detection.class_name,
                timestamp=detection.timestamp,
                image_size=detection.image_size,
            )
            detection.track_id = track_id
            matched_tracks.add(track_id)
        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missing += 1
            if track.missing > self.config.max_missing:
                del self._tracks[track_id]
        return detections

    def predict(self, timestamp: float) -> list[Detection2D]:
        predictions: list[Detection2D] = []
        for track in self._tracks.values():
            bbox = track.predicted_bbox(timestamp).astype(np.float32)
            centroid = (bbox[:2] + bbox[2:]) * 0.5
            predictions.append(
                Detection2D(
                    bbox_xyxy=bbox,
                    class_id=-1,
                    class_name=track.class_name,
                    confidence=max(0.1, 0.8 ** (track.missing + 1)),
                    centroid_xy=centroid,
                    timestamp=timestamp,
                    track_id=track.track_id,
                    track_hits=track.hits,
                    velocity_xy=track.velocity.copy(),
                    image_size=track.image_size,
                )
            )
        return predictions
