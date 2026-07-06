from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from social_bev.types import Detection, Track


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def bbox_to_measurement(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float64)


def measurement_to_bbox(measurement: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = measurement[:4]
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return (float(cx - w * 0.5), float(cy - h * 0.5), float(cx + w * 0.5), float(cy + h * 0.5))


@dataclass
class _TrackState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    confidence: float
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    last_timestamp: float = 0.0
    history: deque[tuple[float, float]] = field(default_factory=deque)
    position_confidence: float = 1.0
    image_ground_point: tuple[float, float] | None = None
    bev_position: tuple[float, float] | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return measurement_to_bbox(self.state[:4])


class MultiObjectTracker:
    """Lightweight CPU multi-person tracker based on Kalman filtering and Hungarian matching."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.maximum_missed_frames = int(config.get("maximum_missed_frames", 15))
        self.minimum_hits = int(config.get("minimum_hits", 3))
        self.history_length = int(config.get("history_length", 30))
        self.iou_weight = float(config.get("iou_weight", 0.6))
        self.distance_weight = float(config.get("distance_weight", 0.4))
        self.max_center_distance_px = float(config.get("max_center_distance_px", 120))
        self._tracks: list[_TrackState] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        predict_only: bool = False,
    ) -> list[Track]:
        people = [d for d in detections if d.category == "person"]
        dt = self._compute_dt(timestamp)
        for track in self._tracks:
            self._predict(track, dt)
            track.age += 1

        if predict_only:
            for track in self._tracks:
                track.missed_frames += 1
            self._prune()
            return self._public_tracks()

        matches, unmatched_tracks, unmatched_detections = self._match(people)
        for track_idx, detection_idx in matches:
            self._update_track(self._tracks[track_idx], people[detection_idx], timestamp)
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].missed_frames += 1
        for detection_idx in unmatched_detections:
            self._create_track(people[detection_idx], timestamp)
        self._prune()
        return self._public_tracks()

    def set_track_projection(
        self,
        track_id: int,
        image_ground_point: tuple[float, float] | None,
        bev_position: tuple[float, float] | None,
        position_confidence: float,
    ) -> None:
        for track in self._tracks:
            if track.track_id != track_id:
                continue
            had_bev_history = track.bev_position is not None
            track.image_ground_point = image_ground_point
            track.bev_position = bev_position
            track.position_confidence = position_confidence
            if bev_position is not None:
                if not had_bev_history and len(track.history) > 0:
                    track.history.clear()
                track.history.append(bev_position)
                while len(track.history) > self.history_length:
                    track.history.popleft()
            return

    def tracks(self) -> list[Track]:
        return self._public_tracks()

    def _compute_dt(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return 1.0
        dt = max(1e-3, float(timestamp) - float(self._last_timestamp))
        self._last_timestamp = timestamp
        return dt

    def _create_track(self, detection: Detection, timestamp: float) -> None:
        measurement = bbox_to_measurement(detection.bbox)
        state = np.zeros(8, dtype=np.float64)
        state[:4] = measurement
        covariance = np.eye(8, dtype=np.float64) * 10.0
        covariance[4:, 4:] *= 100.0
        ground_point = detection.bottom_center
        history: deque[tuple[float, float]] = deque(maxlen=self.history_length)
        history.append(ground_point)
        track = _TrackState(
            track_id=self._next_id,
            state=state,
            covariance=covariance,
            confidence=detection.confidence,
            last_timestamp=timestamp,
            history=history,
            image_ground_point=ground_point,
        )
        self._next_id += 1
        self._tracks.append(track)

    def _predict(self, track: _TrackState, dt: float) -> None:
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt
        process_noise = np.eye(8, dtype=np.float64)
        process_noise[:4, :4] *= 2.0
        process_noise[4:, 4:] *= 8.0
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise

    def _update_track(self, track: _TrackState, detection: Detection, timestamp: float) -> None:
        measurement = bbox_to_measurement(detection.bbox)
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        measurement_noise = np.eye(4, dtype=np.float64) * 10.0
        innovation = measurement - observation @ track.state
        innovation_cov = observation @ track.covariance @ observation.T + measurement_noise
        gain = track.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        track.state = track.state + gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        track.covariance = (identity - gain @ observation) @ track.covariance
        track.confidence = detection.confidence
        track.hits += 1
        track.missed_frames = 0
        track.last_timestamp = timestamp
        ground_point = detection.bottom_center
        track.image_ground_point = ground_point
        if track.bev_position is None:
            track.history.append(ground_point)
            while len(track.history) > self.history_length:
                track.history.popleft()

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self._tracks))), []

        cost = np.zeros((len(self._tracks), len(detections)), dtype=np.float64)
        invalid = np.zeros_like(cost, dtype=bool)
        for i, track in enumerate(self._tracks):
            track_bbox = track.bbox
            track_center = bbox_to_measurement(track_bbox)[:2]
            for j, detection in enumerate(detections):
                det_center = bbox_to_measurement(detection.bbox)[:2]
                iou_score = bbox_iou(track_bbox, detection.bbox)
                distance = float(np.linalg.norm(track_center - det_center))
                distance_score = min(1.0, distance / max(1.0, self.max_center_distance_px))
                cost[i, j] = self.iou_weight * (1.0 - iou_score) + self.distance_weight * distance_score
                invalid[i, j] = iou_score < 0.01 and distance > self.max_center_distance_px

        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, col in zip(row_ind, col_ind):
            if invalid[row, col] or cost[row, col] > 0.95:
                continue
            matches.append((int(row), int(col)))
            matched_tracks.add(int(row))
            matched_detections.add(int(col))
        unmatched_tracks = [idx for idx in range(len(self._tracks)) if idx not in matched_tracks]
        unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _prune(self) -> None:
        self._tracks = [t for t in self._tracks if t.missed_frames <= self.maximum_missed_frames]

    def _public_tracks(self) -> list[Track]:
        public: list[Track] = []
        for track in self._tracks:
            if track.hits < self.minimum_hits and track.missed_frames > 0:
                continue
            if track.hits < self.minimum_hits and track.age > self.minimum_hits:
                continue
            velocity = (float(track.state[4]), float(track.state[5]))
            if track.bev_position is not None and len(track.history) > 0:
                trajectory = list(track.history)
            else:
                trajectory = list(track.history)
            public.append(
                Track(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    image_ground_point=track.image_ground_point,
                    bev_position=track.bev_position,
                    velocity=velocity,
                    age=track.age,
                    missed_frames=track.missed_frames,
                    trajectory=trajectory[-self.history_length :],
                    position_confidence=track.position_confidence,
                )
            )
        return public
