from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MovementDirection = Literal["stationary", "toward_zone", "away_from_zone", "parallel", "unknown"]
AgentClass = Literal["person", "bicycle", "motorcycle", "car", "bus", "truck", "unknown"]
DirectionLabel = Literal[
    "toward_camera",
    "away_from_camera",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
    "stationary",
    "uncertain",
]
PathRelation = Literal[
    "along_path",
    "crossing_path",
    "entering_path",
    "leaving_path",
    "parallel_to_path",
    "uncertain",
]
RiskLevel = Literal["low", "warning", "critical", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Point2D(StrictModel):
    x: float
    y: float


class FramePacket(StrictModel):
    source_type: Literal["huggingface", "local_video", "image_sequence", "ros2"]
    dataset_name: str = ""
    dataset_revision: str = ""
    split: str = ""
    scenario_id: str = ""
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    original_timestamp: float | str | None = None
    fps: float | None = Field(default=None, gt=0)
    image_width: int = Field(ge=0)
    image_height: int = Field(ge=0)
    image_reference: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(StrictModel):
    dataset_id: str
    name: str
    revision: str
    license: str | None = None
    source_url: str
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioInfo(StrictModel):
    scenario_id: str
    dataset_name: str
    dataset_revision: str
    split: str = "prompts"
    frame_count: int = Field(ge=0)
    duration_sec: float = Field(ge=0.0)
    image_width: int = Field(ge=0)
    image_height: int = Field(ge=0)
    first_frame_index: int = 0
    last_frame_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlaybackState(StrictModel):
    scenario_id: str | None = None
    status: Literal["idle", "playing", "paused", "stopped", "ended"] = "idle"
    frame_index: int = Field(default=0, ge=0)
    timestamp_sec: float = Field(default=0.0, ge=0.0)
    speed: float = Field(default=1.0, gt=0)
    loop: bool = False
    step_mode: bool = False
    realtime_mode: bool = True
    experiment_mode: bool = False
    total_frames: int = Field(default=0, ge=0)
    duration_sec: float = Field(default=0.0, ge=0.0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoboPointResult(StrictModel):
    points_normalized: list[Point2D] = Field(default_factory=list)
    points_pixel: list[Point2D] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_response: str = ""
    latency_ms: int = Field(default=0, ge=0)
    model: str = ""
    revision: str = ""


class ZoneDefinition(StrictModel):
    zone_id: str
    scenario_id: str
    name: str
    source: Literal[
        "robopoint",
        "robopoint_sam3",
        "manual",
        "manual_fallback",
        "cached",
        "dataset",
        "imported",
    ]
    coordinate_type: Literal["image", "bev", "robot", "map"] = "image"
    polygon: list[Point2D] = Field(default_factory=list)
    mask_rle: dict[str, Any] | None = None
    prompt: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_frame_index: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    locked: bool = False
    opacity: float = Field(default=0.28, ge=0.0, le=1.0)
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def polygon_requires_three_points(cls, value: list[Point2D]) -> list[Point2D]:
        if value and len(value) < 3:
            raise ValueError("zone polygon must contain at least three points")
        return value


class MotionEstimate(StrictModel):
    track_id: int
    timestamp_sec: float = Field(default=0.0, ge=0.0)
    velocity_vector: tuple[float, float] | None = None
    speed: float = Field(default=0.0, ge=0.0)
    speed_unit: Literal["m/s", "normalized/s", "px/s"] = "normalized/s"
    direction_angle_deg: float = 0.0
    direction_label_geometry: DirectionLabel = "uncertain"
    direction_label_vqa: DirectionLabel = "uncertain"
    direction_label_fused: DirectionLabel = "uncertain"
    is_approximate: bool = True
    velocity_px_per_sec: Point2D | None = None
    speed_px_per_sec: float | None = Field(default=None, ge=0.0)
    movement_direction: MovementDirection = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TrackObservation(StrictModel):
    track_id: int
    class_name: AgentClass = "person"
    timestamp_sec: float = Field(ge=0.0)
    frame_index: int = Field(default=0, ge=0)
    mask_rle: dict[str, Any] | None = None
    mask_polygon: list[Point2D] = Field(default_factory=list)
    bounding_box: tuple[float, float, float, float] | None = None
    centroid_image: Point2D | None = None
    ground_contact_point: Point2D | None = None
    centroid: Point2D
    bottom_center: Point2D | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    track_age_sec: float = Field(default=0.0, ge=0.0)
    lost_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrackHistory(StrictModel):
    track_id: int
    observations: list[TrackObservation] = Field(default_factory=list)
    first_seen_timestamp: float | None = None
    last_seen_timestamp: float | None = None
    age: int = Field(default=0, ge=0)
    lost_count: int = Field(default=0, ge=0)


class GeometryPrediction(StrictModel):
    track_id: int
    prediction_method: Literal["constant_velocity"] = "constant_velocity"
    distance_to_zone: float | None = Field(default=None, ge=0.0)
    time_to_zone_sec: float | None = Field(default=None, ge=0.0)
    zone_relation: Literal["outside", "approaching", "inside", "leaving", "unknown"] = "unknown"
    predicted_path: list[Point2D] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VQAQuestion(StrictModel):
    id: str
    category: Literal["spatial", "spatiotemporal", "social"]
    question: str
    allowed_answers: list[str]
    required_inputs: list[str]
    answer_schema: dict[str, Any] = Field(default_factory=dict)
    safety_relevance: str
    trigger_conditions: list[str] = Field(default_factory=list)


class VQARequest(StrictModel):
    scenario_id: str
    current_frame_index: int = Field(ge=0)
    frame_indices: list[int]
    timestamps: list[float]
    question_id: str
    category: str
    question: str
    track_metadata: list[dict[str, Any]] = Field(default_factory=list)
    zone_metadata: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""
    revision: str = ""


class VQAResult(StrictModel):
    scenario_id: str
    current_frame_index: int = Field(ge=0)
    frame_indices: list[int]
    timestamps: list[float]
    question_id: str
    category: str
    question: str
    track_metadata: list[dict[str, Any]] = Field(default_factory=list)
    zone_metadata: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    revision: str = ""
    raw_response: str = ""
    parsed_response: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(default=0, ge=0)
    parse_valid: bool = False
    retry_count: int = Field(default=0, ge=0)


class RoadSegmentationResult(StrictModel):
    scenario_id: str = ""
    frame_index: int = Field(default=0, ge=0)
    timestamp_sec: float = Field(default=0.0, ge=0.0)
    source: Literal[
        "robopoint_sam3",
        "manual_fallback",
        "cached",
        "fixture_color_segmentation",
        "unavailable",
    ] = "unavailable"
    mask_rle: dict[str, Any] | None = None
    polygon: list[Point2D] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    prompt: str = ""
    is_valid: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VQADirectionEstimate(StrictModel):
    track_id: int
    motion_state: Literal["moving", "stationary", "uncertain"] = "uncertain"
    direction_label: DirectionLabel = "uncertain"
    path_relation: PathRelation = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    updated_at_sec: float | None = Field(default=None, ge=0.0)
    parse_valid: bool = False


class DynamicRiskZone(StrictModel):
    track_id: int
    class_name: AgentClass = "unknown"
    timestamp_sec: float = Field(default=0.0, ge=0.0)
    prediction_horizon_sec: float = Field(default=3.0, gt=0.0)
    predicted_points: list[Point2D] = Field(default_factory=list)
    risk_polygon: list[Point2D] = Field(default_factory=list)
    speed: float = Field(default=0.0, ge=0.0)
    direction: DirectionLabel = "uncertain"
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    intersects_robot_corridor: bool = False
    risk_level: RiskLevel = "unknown"
    time_to_intersection_sec: float | None = Field(default=None, ge=0.0)


class RobotCorridor(StrictModel):
    scenario_id: str = ""
    timestamp_sec: float = Field(default=0.0, ge=0.0)
    polygon: list[Point2D] = Field(default_factory=list)
    origin: Point2D | None = None
    heading_vector: Point2D | None = None
    coordinate_type: Literal["metric", "approximate_camera_relative"] = (
        "approximate_camera_relative"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_approximate: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisSystemStatus(StrictModel):
    tracking_fps: float = Field(default=0.0, ge=0.0)
    vqa_update_interval_sec: float = Field(default=2.0, gt=0.0)
    analysis_delay_ms: int = Field(default=0, ge=0)
    analysis_age_ms: int = Field(default=0, ge=0)
    vqa_last_update_sec: float = Field(default=0.0, ge=0.0)
    tracking_status: Literal["ok", "degraded", "unavailable"] = "unavailable"
    road_status: Literal["ok", "degraded", "unavailable"] = "unavailable"
    vqa_status: Literal["ok", "degraded", "unavailable"] = "unavailable"
    message: str = ""


class AnalysisPacket(StrictModel):
    scenario_id: str
    video_timestamp_sec: float = Field(ge=0.0)
    analysis_timestamp_sec: float = Field(ge=0.0)
    road: RoadSegmentationResult
    tracks: list[TrackObservation] = Field(default_factory=list)
    motions: list[MotionEstimate] = Field(default_factory=list)
    vqa_directions: list[VQADirectionEstimate] = Field(default_factory=list)
    risk_zones: list[DynamicRiskZone] = Field(default_factory=list)
    robot_corridor: RobotCorridor
    system_status: AnalysisSystemStatus = Field(default_factory=AnalysisSystemStatus)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SafetyDecision(StrictModel):
    timestamp_sec: float = Field(ge=0.0)
    scenario_id: str
    frame_index: int = Field(ge=0)
    zone_state: Literal["clear", "occupied", "approaching", "leaving", "unknown"] = "unknown"
    risk_level: Literal["safe", "warning", "critical", "unknown"] = "unknown"
    recommended_action: Literal[
        "continue", "slow_down", "yield", "pause", "resume", "human_review"
    ] = "human_review"
    target_track_ids: list[int] = Field(default_factory=list)
    time_to_zone_sec: float | None = Field(default=None, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    hard_rule_triggered: bool = False
    hard_rule_reason: str = ""
    vqa_overridden: bool = False
    source: dict[str, bool] = Field(
        default_factory=lambda: {"tracker": False, "geometry": False, "vqa": False}
    )


class DecisionEvent(StrictModel):
    event_id: str
    event_type: str
    scenario_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    decision: SafetyDecision | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(StrictModel):
    run_id: str
    dataset_id: str
    dataset_revision: str
    scenarios: list[str] = Field(default_factory=list)
    time_steps: list[int] = Field(default_factory=lambda: [1, 3, 5, 8])
    sampling_intervals_sec: list[float] = Field(default_factory=lambda: [0.2, 0.5, 1.0])
    prediction_horizons_sec: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    zone_sources: list[str] = Field(default_factory=lambda: ["manual"])
    vqa_models: list[str] = Field(default_factory=list)
    formal: bool = False


class ExperimentRecord(StrictModel):
    run_id: str
    scenario_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    prediction: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceHealth(StrictModel):
    service: str
    status: Literal["ok", "degraded", "error"]
    version: str = "0.1.0"
    ready: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(StrictModel):
    service: str
    model: str
    revision: str
    provider: str = ""
    loaded: bool = False
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeEnvironment(StrictModel):
    os: str
    python_version: str
    cpu: str = ""
    gpu: str | None = None
    cuda: str | None = None
    jetpack_l4t: str | None = None
    ram_gb: float | None = None
    vram_gb: float | None = None
    git_commit: str | None = None
    container_digest: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
