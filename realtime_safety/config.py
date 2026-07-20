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
    confidence: float = 0.3
    iou: float = 0.5
    frequency_hz: float = 10.0
    fp16: bool = True


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
    input_size: int = 224
    frequency_hz: float = 2.0
    fast_depth_frequency_hz: float = 10.0
    max_points: int = 30_000
    voxel_size: float = 0.08
    confidence_threshold: float = 0.25
    display_confidence_threshold: float = 0.0
    filter_sky: bool = False
    max_relative_depth: float = 20.0
    anchor_interval: int = 30
    fp16: bool = True


@dataclass(slots=True)
class TrackingConfig:
    max_missing: int = 12
    visual_hold_updates: int = 2
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
    max_video_width: int = 960
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
