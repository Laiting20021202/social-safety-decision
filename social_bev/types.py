from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """A decoded BGR frame with source metadata."""

    index: int
    timestamp: float
    image: np.ndarray
    path: str | None = None


@dataclass(slots=True)
class SegmentationResult:
    """Semantic segmentation output reduced to a binary walkable mask."""

    mask: np.ndarray
    raw_labels: np.ndarray | None
    confidence: np.ndarray | None
    processing_ms: float
    backend: str
    labels_present: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Detection:
    """Object detection in image coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    category: str

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)


@dataclass(slots=True)
class ObstacleRegion:
    """Estimated obstacle region from known detections or unknown RGB cues."""

    bbox: tuple[float, float, float, float]
    contour: np.ndarray
    confidence: float
    label: str
    category: str
    ground_points: list[tuple[float, float]] = field(default_factory=list)
    bev_points: list[tuple[float, float]] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class Track:
    """A CPU-tracked person with bounded image and BEV trajectory history."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    image_ground_point: tuple[float, float] | None
    bev_position: tuple[float, float] | None
    velocity: tuple[float, float]
    age: int
    missed_frames: int
    trajectory: list[tuple[float, float]]
    position_confidence: float = 1.0


@dataclass(slots=True)
class Calibration:
    """Image-to-ground-plane homography and BEV metadata."""

    homography: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    metric_bev: bool
    bev_config: dict[str, float | int]
    label: str = "METRIC BEV"


@dataclass(slots=True)
class BEVResult:
    """Rendered BEV and occupancy grid for one frame."""

    image: np.ndarray
    occupancy_grid: np.ndarray
    metric_bev: bool
    label: str
    layers: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    """Complete perception output for a single frame."""

    frame_index: int
    timestamp: float
    annotated_frame: np.ndarray
    visualization: np.ndarray
    walkable_mask: np.ndarray
    detections: list[Detection]
    tracks: list[Track]
    unknown_obstacles: list[ObstacleRegion]
    bev: BEVResult
    processing_ms: dict[str, float]
    fps: float
    average_fps: float

    def to_json_dict(self) -> dict[str, Any]:
        known_obstacles = [d for d in self.detections if d.category == "known_obstacle"]
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "metric_bev": self.bev.metric_bev,
            "people": [_track_to_dict(t) for t in self.tracks],
            "known_obstacles": [_detection_to_dict(d) for d in known_obstacles],
            "unknown_obstacles": [_obstacle_to_dict(o) for o in self.unknown_obstacles],
            "processing_ms": {k: round(float(v), 3) for k, v in self.processing_ms.items()},
            "fps": round(float(self.fps), 3),
            "average_fps": round(float(self.average_fps), 3),
        }


def _round_pair(point: tuple[float, float] | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": _bbox_to_list(detection.bbox),
        "confidence": round(float(detection.confidence), 4),
        "class_id": int(detection.class_id),
        "class_name": detection.class_name,
        "category": detection.category,
        "ground_point": _round_pair(detection.bottom_center),
    }


def _track_to_dict(track: Track) -> dict[str, Any]:
    return {
        "track_id": int(track.track_id),
        "bbox": _bbox_to_list(track.bbox),
        "confidence": round(float(track.confidence), 4),
        "image_ground_point": _round_pair(track.image_ground_point),
        "bev_position": _round_pair(track.bev_position),
        "velocity": _round_pair(track.velocity),
        "age": int(track.age),
        "missed_frames": int(track.missed_frames),
        "position_confidence": round(float(track.position_confidence), 4),
        "trajectory": [_round_pair(p) for p in track.trajectory],
    }


def _obstacle_to_dict(obstacle: ObstacleRegion) -> dict[str, Any]:
    return {
        "bbox": _bbox_to_list(obstacle.bbox),
        "confidence": round(float(obstacle.confidence), 4),
        "label": obstacle.label,
        "category": obstacle.category,
        "ground_points": [_round_pair(p) for p in obstacle.ground_points],
        "bev_points": [_round_pair(p) for p in obstacle.bev_points],
        "note": obstacle.note,
    }


PathLike = str | Path

