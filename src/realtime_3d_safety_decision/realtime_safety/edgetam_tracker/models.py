from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrackingState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    DELETED = "DELETED"


class PointCloudQuality(str, Enum):
    GOOD = "GOOD"
    SPARSE = "SPARSE"
    INVALID = "INVALID"


class MaskQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(slots=True)
class AABB:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        self.minimum = np.asarray(self.minimum, dtype=np.float32).reshape(3)
        self.maximum = np.asarray(self.maximum, dtype=np.float32).reshape(3)

    @property
    def center(self) -> np.ndarray:
        return ((self.minimum + self.maximum) * 0.5).astype(np.float32)

    @property
    def size(self) -> np.ndarray:
        return np.maximum(self.maximum - self.minimum, 0.0).astype(np.float32)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def translated(self, offset: np.ndarray) -> "AABB":
        shift = np.asarray(offset, dtype=np.float32).reshape(3)
        return AABB(self.minimum + shift, self.maximum + shift)

    def expanded(self, margin: float | np.ndarray) -> "AABB":
        padding = np.broadcast_to(np.asarray(margin, dtype=np.float32), (3,))
        return AABB(self.minimum - padding, self.maximum + padding)

    def iou(self, other: "AABB") -> float:
        overlap = np.maximum(
            np.minimum(self.maximum, other.maximum)
            - np.maximum(self.minimum, other.minimum),
            0.0,
        )
        intersection = float(np.prod(overlap))
        union = self.volume + other.volume - intersection
        return 0.0 if union <= 1e-12 else intersection / union


@dataclass(slots=True)
class OBB:
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.size = np.maximum(
            np.asarray(self.size, dtype=np.float32).reshape(3), 0.0
        )
        self.rotation = np.asarray(self.rotation, dtype=np.float32).reshape(3, 3)

    def translated(self, offset: np.ndarray) -> "OBB":
        return OBB(self.center + np.asarray(offset, dtype=np.float32), self.size, self.rotation)


@dataclass(slots=True)
class CloudFrame:
    """A finite point cloud with optional RGB and image-pixel correspondence."""

    points: np.ndarray
    stamp: float
    frame_id: str
    colors: np.ndarray | None = None
    pixels_uv: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    source_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        count = len(self.points)
        if self.colors is not None:
            colors = np.asarray(self.colors)
            if colors.ndim == 1:
                colors = np.repeat(colors.reshape(-1, 1), 3, axis=1)
            self.colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[:count]
        if self.pixels_uv is not None:
            self.pixels_uv = np.asarray(self.pixels_uv, dtype=np.int32).reshape(-1, 2)[:count]
        if self.source_indices is None:
            self.source_indices = np.arange(count, dtype=np.int64)
        else:
            self.source_indices = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)[:count]

    def select(self, selector: np.ndarray) -> "CloudFrame":
        selector = np.asarray(selector)
        return CloudFrame(
            points=self.points[selector],
            colors=None if self.colors is None else self.colors[selector],
            pixels_uv=None if self.pixels_uv is None else self.pixels_uv[selector],
            source_indices=self.source_indices[selector],
            stamp=self.stamp,
            frame_id=self.frame_id,
            image_shape=self.image_shape,
        )


@dataclass(slots=True)
class Cluster3D:
    cluster_id: int
    points: np.ndarray
    centroid: np.ndarray
    median_center: np.ndarray
    aabb: AABB
    obb: OBB
    nearest_point: np.ndarray
    nearest_distance: float
    point_count: int
    colors: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    pixels_uv: np.ndarray | None = None
    density: float = 0.0
    depth_variance: float = 0.0
    missing_depth_ratio: float = 0.0
    quality: PointCloudQuality = PointCloudQuality.GOOD
    quality_score: float = 1.0
    innovation_distance: float = 0.0
    projection_mask: np.ndarray | None = None
    projection_bbox: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        self.centroid = np.asarray(self.centroid, dtype=np.float32).reshape(3)
        self.median_center = np.asarray(self.median_center, dtype=np.float32).reshape(3)
        self.nearest_point = np.asarray(self.nearest_point, dtype=np.float32).reshape(3)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        if self.source_indices is not None:
            self.source_indices = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)
        if self.pixels_uv is not None:
            self.pixels_uv = np.asarray(self.pixels_uv, dtype=np.int32).reshape(-1, 2)


@dataclass(slots=True)
class ProjectionPrompt:
    track_id: int
    frame_index: int
    box_xyxy: np.ndarray
    positive_points: np.ndarray
    negative_points: np.ndarray | None = None
    projection_mask: np.ndarray | None = None
    re_prompt: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        self.box_xyxy = np.asarray(self.box_xyxy, dtype=np.float32).reshape(4)
        self.positive_points = np.asarray(
            self.positive_points, dtype=np.float32
        ).reshape(-1, 2)
        if self.negative_points is not None:
            self.negative_points = np.asarray(
                self.negative_points, dtype=np.float32
            ).reshape(-1, 2)


@dataclass(slots=True)
class MaskObservation:
    track_id: int
    frame_index: int
    stamp: float
    mask: np.ndarray
    model_score: float | None = None
    quality: MaskQuality = MaskQuality.UNAVAILABLE
    quality_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    re_prompted: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)


@dataclass(slots=True)
class TrackEstimate:
    track_id: int
    state: TrackingState
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    aabb: AABB
    obb: OBB
    nearest_point: np.ndarray
    nearest_distance: float
    point_count: int
    hit_count: int
    missed_count: int
    age_frames: int
    first_timestamp: float
    last_measurement_timestamp: float
    filter_timestamp: float
    confidence: float
    pointcloud_quality: PointCloudQuality
    mask_quality: MaskQuality = MaskQuality.UNAVAILABLE
    mask_quality_score: float = 0.0
    source_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    source_colors: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    predicted_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    uncertainty_margin: float = 0.0
    last_association_cost: float = 0.0
    edge_tam_refined: bool = False
    semantic_class: str = ""
    semantic_confirmed: bool = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float32).reshape(3)
        self.covariance = np.asarray(self.covariance, dtype=np.float64).reshape(6, 6)
        self.nearest_point = np.asarray(self.nearest_point, dtype=np.float32).reshape(3)
        self.source_points = np.asarray(self.source_points, dtype=np.float32).reshape(-1, 3)
        if self.source_colors is not None:
            self.source_colors = np.asarray(self.source_colors, dtype=np.uint8).reshape(-1, 3)
        if self.source_indices is not None:
            self.source_indices = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)
        self.predicted_positions = np.asarray(
            self.predicted_positions, dtype=np.float32
        ).reshape(-1, 3)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


@dataclass(slots=True)
class PipelineTiming:
    preprocessing_ms: float = 0.0
    clustering_ms: float = 0.0
    tracking_ms: float = 0.0
    edgetam_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0
