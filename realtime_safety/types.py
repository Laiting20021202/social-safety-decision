from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SafetyLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    STOP = "STOP"
    DEGRADED = "DEGRADED"


class RecommendedAction(str, Enum):
    CONTINUE = "CONTINUE"
    SLOW_DOWN = "SLOW_DOWN"
    STOP = "STOP"
    DETOUR_LEFT = "DETOUR_LEFT"
    DETOUR_RIGHT = "DETOUR_RIGHT"
    WAIT = "WAIT"


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    source_timestamp: float
    capture_timestamp: float
    bgr: np.ndarray
    rgb: np.ndarray
    original_fps: float
    original_width: int
    original_height: int


@dataclass(slots=True)
class Detection2D:
    bbox_xyxy: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    centroid_xy: np.ndarray
    timestamp: float
    mask: np.ndarray | None = None
    track_id: int | None = None
    track_hits: int = 1
    track_missing: int = 0
    is_prediction: bool = False
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    image_size: tuple[int, int] | None = None


@dataclass(slots=True)
class PointCloudFrame:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    pointmap: np.ndarray
    frame_index: int
    timestamp: float
    anchor_frame_index: int
    inference_ms: float
    valid: bool
    source: str
    tracking_points: np.ndarray | None = None
    camera_transform: np.ndarray | None = None
    dense_confidence: np.ndarray | None = None
    metric_scale: float | None = None
    reference_depth_m: float | None = None
    reference_observed_depth: float | None = None
    apriltag_locked: bool = False
    apriltag_id: int | None = None
    apriltag_size_m: float | None = None
    apriltag_observed_edge_m: float | None = None
    apriltag_scale_correction: float | None = None
    apriltag_center_xyz: np.ndarray | None = None
    apriltag_corners_xyz: np.ndarray | None = None
    apriltag_age_frames: int | None = None


@dataclass(slots=True)
class RobotArmState:
    """Camera-aligned estimate of the visible Koch arm."""

    center_xyz: np.ndarray
    center_xy: np.ndarray
    image_size: tuple[int, int]
    mask_pixels: int
    point_count: int
    confidence: float
    timestamp: float
    held_frames: int = 0
    localization_source: str = "rgb_depth"
    # Optional named FK points (for example left_tcp/right_tcp).  The legacy
    # center remains the midpoint so existing consumers stay compatible.
    link_points_xyz: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class BBox3D:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum


@dataclass(slots=True)
class ObstacleObservation3D:
    track_id: int
    class_name: str
    confidence: float
    position_xyz: np.ndarray
    bbox3d: BBox3D
    radius: float
    point_count: int
    timestamp: float
    points: np.ndarray | None = None


@dataclass(slots=True)
class Track3DState:
    track_id: int
    class_name: str
    position_xyz: np.ndarray
    velocity_xyz: np.ndarray
    acceleration_xyz: np.ndarray
    covariance: np.ndarray
    bbox3d: BBox3D
    radius: float
    hit_count: int
    missing_count: int
    last_timestamp: float
    motion_state: str
    confidence: float
    history: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class DangerZone:
    track_id: int
    predicted_positions: np.ndarray
    radii: np.ndarray
    predicted_direction: np.ndarray
    predicted_speed: float
    risk_score: float
    closest_predicted_distance: float
    ttc: float | None
    risk_level: SafetyLevel
    dynamic: bool


@dataclass(slots=True)
class PathCandidate:
    points: np.ndarray
    safe: bool
    score: float
    name: str
    collision_point: np.ndarray | None = None


@dataclass(slots=True)
class PerformanceSnapshot:
    input_fps: float = 0.0
    display_fps: float = 0.0
    segmentation_fps: float = 0.0
    reconstruction_fps: float = 0.0
    safety_fps: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    dropped_frames: int = 0
    queue_size: int = 0
    queue_capacity: int = 0
    ram_mb: float = 0.0
    vram_used_mb: float = 0.0


@dataclass(slots=True)
class SafetySnapshot:
    timestamp: float
    frame_index: int
    safety_state: SafetyLevel
    recommended_action: RecommendedAction
    tracks: list[Track3DState]
    danger_zones: list[DangerZone]
    candidates: list[PathCandidate]
    selected_path: PathCandidate | None
    metric_valid: bool
    degraded_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineSnapshot:
    frame: FramePacket | None = None
    annotated_bgr: np.ndarray | None = None
    detections: list[Detection2D] = field(default_factory=list)
    pointcloud: PointCloudFrame | None = None
    robot_arm: RobotArmState | None = None
    people: list[Track3DState] = field(default_factory=list)
    safety: SafetySnapshot | None = None
    performance: PerformanceSnapshot = field(default_factory=PerformanceSnapshot)
    profile: str = "AUTO"
    depth_mode: str = "fast_depth"
    scale_mode: str = "relative"
    status: dict[str, Any] = field(default_factory=dict)
