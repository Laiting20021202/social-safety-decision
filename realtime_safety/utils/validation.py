from __future__ import annotations

from realtime_safety.config import AppConfig


def validate_config(config: AppConfig) -> None:
    if config.mode not in {"safety", "reconstruction"}:
        raise ValueError(f"Unsupported application mode: {config.mode}")
    if config.video.queue_size < 1:
        raise ValueError("video.queue_size must be positive")
    if config.safety.target_hz <= 0 or config.segmentation.frequency_hz <= 0:
        raise ValueError("update frequencies must be positive")
    if config.reconstruction.max_points < 100:
        raise ValueError("reconstruction.max_points must be at least 100")
    if config.reconstruction.display_confidence_threshold < 0:
        raise ValueError("reconstruction.display_confidence_threshold cannot be negative")
    if config.reconstruction.anchor_interval < 0:
        raise ValueError("reconstruction.anchor_interval cannot be negative")
    if config.tracking.visual_hold_updates < 0:
        raise ValueError("tracking.visual_hold_updates cannot be negative")
    if config.gui.history_frames < 1 or config.gui.history_stride < 1:
        raise ValueError("GUI history settings must be positive")
    if config.scale_mode not in {"relative", "calibrated", "rgbd"}:
        raise ValueError(f"Unsupported scale mode: {config.scale_mode}")
    if config.scale_mode == "calibrated" and config.manual_scale is not None and config.manual_scale <= 0:
        raise ValueError("manual_scale must be positive")
