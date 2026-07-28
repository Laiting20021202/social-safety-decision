from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class VideoConfig:
    queue_size: int = 2
    loop: bool = False
    playback_speed: float = 1.0


@dataclass(slots=True)
class SegmentationConfig:
    model: str = "yolo11n-seg.pt"
    input_size: int = 320
    # Confidence used to create a new, visible detection.
    confidence: float = 0.3
    # Lower detector floor used by a temporal tracker to recover an existing
    # person through blur or partial occlusion.
    tracking_confidence: float = 0.12
    tracker_config: str | None = None
    iou: float = 0.5
    frequency_hz: float = 10.0
    fp16: bool = True
    # Suppress detections caused by the camera-mounted robot itself.  The
    # filter follows the robot's distinctive green links from a small,
    # camera-fixed base ROI instead of excluding a fixed workspace rectangle.
    robot_self_filter: bool = False
    robot_anchor_roi: tuple[float, float, float, float] = (0.38, 0.0, 0.64, 0.28)
    robot_green_hsv_lower: tuple[int, int, int] = (40, 55, 25)
    robot_green_hsv_upper: tuple[int, int, int] = (95, 255, 255)
    robot_mask_dilation_px: int = 14
    robot_tip_extension_px: int = 34
    robot_mask_hold_frames: int = 3
    robot_mask_temporal_frames: int = 3
    robot_component_link_px: int = 18
    robot_reject_overlap: float = 0.65
    robot_reject_min_overlap: float = 0.25
    robot_min_residual_pixels: int = 80
    robot_center_ema_alpha: float = 0.25
    robot_center_hold_frames: int = 12


@dataclass(slots=True)
class ReconstructionConfig:
    depth_mode: str = "hybrid"
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    st4rtrack_path: str | None = None
    st4rtrack_checkpoint: str | None = None
    video_depth_path: str | None = None
    video_depth_checkpoint: str | None = None
    focal_length_x: float | None = None
    focal_length_y: float | None = None
    principal_point_x: float | None = None
    principal_point_y: float | None = None
    metric_reference_depth_m: float | None = None
    # Normalized [x_min, y_min, x_max, y_max] around a fixed reference object.
    metric_reference_roi: tuple[float, float, float, float] = (0.45, 0.35, 0.55, 0.65)
    # A foreground percentile is more robust than the median when the ROI
    # contains a small fixed object in front of the background.
    metric_reference_percentile: float = 20.0
    metric_reference_warmup_frames: int = 8
    metric_reference_ema_alpha: float = 0.08
    input_size: int = 224
    frequency_hz: float = 2.0
    fast_depth_frequency_hz: float = 10.0
    max_points: int = 30_000
    voxel_size: float = 0.08
    confidence_threshold: float = 0.25
    display_confidence_threshold: float = 0.0
    filter_sky: bool = False
    # Reject outliers in the model's native output before metric calibration.
    max_relative_depth: float = 20.0
    # Optional operational range after metric calibration.
    max_metric_depth_m: float | None = None
    anchor_interval: int = 30
    fp16: bool = True


@dataclass(slots=True)
class TrackingConfig:
    max_missing: int = 12
    visual_hold_updates: int = 2
    # A conservative ROS obstacle cloud may outlive the GUI prediction briefly
    # so a downstream avoidance controller does not see obstacle/no-obstacle
    # flicker on isolated segmentation or depth failures.
    obstacle_cloud_hold_updates: int = 12
    confirmation_hits: int = 3
    bbox_smoothing_alpha: float = 0.45
    obstacle_center_max_step_m: float = 0.18
    iou_threshold: float = 0.2
    association_distance: float = 1.5
    dynamic_enter_speed: float = 0.15
    dynamic_exit_speed: float = 0.08
    minimum_dynamic_hits: int = 3


@dataclass(slots=True)
class SafetyConfig:
    target_hz: float = 10.0
    prediction_horizon: float = 3.0
    prediction_timestep: float = 0.2
    robot_radius: float = 0.35
    safety_margin: float = 0.2
    uncertainty_gain: float = 0.25
    stop_ttc: float = 1.0
    warning_ttc: float = 2.5
    stop_clearance: float = 0.4
    warning_clearance: float = 0.8
    release_safe_updates: int = 5


@dataclass(slots=True)
class GuiConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    presentation_mode: bool = False
    max_video_width: int = 960
    video_fps: float = 24.0
    point_size: float = 0.012
    history_frames: int = 64
    history_stride: int = 4


@dataclass(slots=True)
class AppConfig:
    name: str = "realtime_fast"
    mode: str = "safety"
    people_overlay: bool = False
    device: str = "cuda"
    scale_mode: str = "relative"
    camera_height: float | None = None
    manual_scale: float | None = None
    video: VideoConfig = field(default_factory=VideoConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)


_SECTIONS: dict[str, type] = {
    "video": VideoConfig,
    "segmentation": SegmentationConfig,
    "reconstruction": ReconstructionConfig,
    "tracking": TrackingConfig,
    "safety": SafetyConfig,
    "gui": GuiConfig,
}


def _only_known(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    known = cls.__dataclass_fields__
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return values


def load_config(profile: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    candidate = Path(profile)
    if not candidate.exists():
        root = Path(__file__).resolve().parents[1]
        candidate = root / "configs" / f"{profile}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Profile not found: {profile}")
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if overrides:
        raw.update(overrides)
    section_values = {
        name: cls(**_only_known(cls, raw.pop(name, {}) or {}))
        for name, cls in _SECTIONS.items()
    }
    return AppConfig(**_only_known(AppConfig, raw), **section_values)
