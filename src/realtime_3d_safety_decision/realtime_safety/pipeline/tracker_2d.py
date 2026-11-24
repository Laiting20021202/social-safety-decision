from __future__ import annotations

from dataclasses import dataclass, field

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
    confidence: float
    timestamp: float
    image_size: tuple[int, int] | None = None
    class_votes: dict[str, float] = field(default_factory=dict)
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
        predicted = track.predicted_bbox(detection.timestamp)
        iou = bbox_iou(predicted, detection.bbox_xyxy)
        scale = max(np.linalg.norm(predicted[2:] - predicted[:2]), 1.0)
        distance = np.linalg.norm((predicted[:2] + predicted[2:]) * 0.5 - detection.centroid_xy) / scale
        if (
            iou < self.config.iou_threshold
            and distance > self.config.association_distance
        ):
            return float("inf")
        # COCO labels commonly alternate between person/chair/bag around a
        # partially occluded obstacle. A small class-change penalty preserves
        # spatial identity while the vote hysteresis below prevents one noisy
        # label from changing the published obstacle class.
        class_penalty = 0.0 if track.class_name == detection.class_name else 0.15
        normalized_distance = float(
            distance / max(self.config.association_distance, 1e-6)
        )
        return (
            0.70 * (1.0 - iou)
            + 0.30 * min(normalized_distance, 2.0)
            + class_penalty
        )

    def update(self, detections: list[Detection2D], timestamp: float) -> list[Detection2D]:
        track_ids = list(self._tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if track_ids and detections:
            costs = np.array(
                [[self._cost(self._tracks[track_id], det) for det in detections] for track_id in track_ids],
                dtype=np.float32,
            )
            rows, columns = linear_sum_assignment(
                np.where(np.isfinite(costs), costs, 1e6)
            )
            for row, column in zip(rows, columns):
                if (
                    not np.isfinite(costs[row, column])
                    or costs[row, column] > 1.10
                ):
                    continue
                track_id = track_ids[row]
                track = self._tracks[track_id]
                detection = detections[column]
                self._apply_detection(track, detection)
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
                confidence=detection.confidence,
                timestamp=detection.timestamp,
                image_size=detection.image_size,
                class_votes={detection.class_name: float(detection.confidence)},
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

    def update_external(self, detections: list[Detection2D], timestamp: float) -> list[Detection2D]:
        """Maintain hold/prediction state around IDs assigned by ByteTrack."""

        matched_tracks: set[int] = set()
        for detection in detections:
            if detection.track_id is None:
                continue
            track_id = int(detection.track_id)
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = _Track2D(
                    track_id=track_id,
                    bbox=detection.bbox_xyxy.copy(),
                    centroid=detection.centroid_xy.copy(),
                    velocity=np.zeros(2, dtype=np.float32),
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    timestamp=detection.timestamp,
                    image_size=detection.image_size,
                    class_votes={detection.class_name: float(detection.confidence)},
                )
                detection.track_hits = 1
                self._next_id = max(self._next_id, track_id + 1)
            else:
                self._apply_detection(track, detection)
            matched_tracks.add(track_id)
        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missing += 1
            if track.missing > self.config.max_missing:
                del self._tracks[track_id]
        return detections

    def predict_missing(
        self,
        timestamp: float,
        max_missing: int | None = None,
    ) -> list[Detection2D]:
        limit = self.config.max_missing if max_missing is None else min(
            int(max_missing), self.config.max_missing
        )
        return [
            self._prediction(track, timestamp)
            for track in self._tracks.values()
            if 0 < track.missing <= limit
        ]

    def predict(self, timestamp: float) -> list[Detection2D]:
        return [self._prediction(track, timestamp) for track in self._tracks.values()]

    def _apply_detection(self, track: _Track2D, detection: Detection2D) -> None:
        measured_class = detection.class_name
        dt = max(detection.timestamp - track.timestamp, 1e-3)
        measured_velocity = (detection.centroid_xy - track.centroid) / dt
        track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
        predicted_bbox = track.predicted_bbox(detection.timestamp)
        alpha = float(self.config.bbox_smoothing_alpha)
        track.bbox = (
            (1.0 - alpha) * predicted_bbox
            + alpha * detection.bbox_xyxy
        ).astype(np.float32)
        track.centroid = (
            (1.0 - alpha)
            * ((predicted_bbox[:2] + predicted_bbox[2:]) * 0.5)
            + alpha * detection.centroid_xy
        ).astype(np.float32)
        track.confidence = detection.confidence
        track.timestamp = detection.timestamp
        track.missing = 0
        track.hits += 1
        for class_name in tuple(track.class_votes):
            track.class_votes[class_name] *= 0.75
            if track.class_votes[class_name] < 0.01:
                del track.class_votes[class_name]
        track.class_votes[measured_class] = (
            track.class_votes.get(measured_class, 0.0)
            + max(float(detection.confidence), 0.05)
        )
        candidate = max(track.class_votes, key=track.class_votes.get)
        current_score = track.class_votes.get(track.class_name, 0.0)
        candidate_score = track.class_votes[candidate]
        if candidate == track.class_name or candidate_score >= 1.15 * max(
            current_score,
            0.05,
        ):
            track.class_name = candidate
        detection.track_id = track.track_id
        detection.class_name = track.class_name
        detection.track_hits = track.hits
        detection.track_missing = 0
        detection.is_prediction = False
        detection.velocity_xy = track.velocity.astype(np.float32)
        detection.bbox_xyxy = track.bbox.copy()
        detection.centroid_xy = track.centroid.copy()

    @staticmethod
    def _prediction(track: _Track2D, timestamp: float) -> Detection2D:
        bbox = track.predicted_bbox(timestamp).astype(np.float32)
        centroid = (bbox[:2] + bbox[2:]) * 0.5
        return Detection2D(
            bbox_xyxy=bbox,
            class_id=-1,
            class_name=track.class_name,
            confidence=max(0.05, track.confidence * 0.85**track.missing),
            centroid_xy=centroid,
            timestamp=timestamp,
            track_id=track.track_id,
            track_hits=track.hits,
            track_missing=track.missing,
            is_prediction=True,
            velocity_xy=track.velocity.copy(),
            image_size=track.image_size,
        )
