from __future__ import annotations

from realtime_safety.config import AppConfig


def validate_config(config: AppConfig) -> None:
    if config.video.queue_size < 1:
        raise ValueError("video.queue_size must be positive")
    if config.safety.target_hz <= 0 or config.segmentation.frequency_hz <= 0:
        raise ValueError("update frequencies must be positive")
    if config.reconstruction.max_points < 100:
        raise ValueError("reconstruction.max_points must be at least 100")
    if config.scale_mode not in {"relative", "calibrated", "rgbd"}:
        raise ValueError(f"Unsupported scale mode: {config.scale_mode}")
    if config.scale_mode == "calibrated" and config.manual_scale is not None and config.manual_scale <= 0:
        raise ValueError("manual_scale must be positive")
