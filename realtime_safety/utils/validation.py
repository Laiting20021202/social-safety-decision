from __future__ import annotations

from realtime_safety.config import AppConfig


def validate_config(config: AppConfig) -> None:
    if config.mode not in {"safety", "reconstruction"}:
        raise ValueError(f"Unsupported application mode: {config.mode}")
    if config.video.queue_size < 1:
        raise ValueError("video.queue_size must be positive")
    if config.safety.target_hz <= 0 or config.segmentation.frequency_hz <= 0:
        raise ValueError("update frequencies must be positive")
    if not 0.0 < config.segmentation.tracking_confidence <= config.segmentation.confidence <= 1.0:
        raise ValueError(
            "segmentation confidence must satisfy 0 < tracking_confidence <= confidence <= 1"
        )
    robot_roi = config.segmentation.robot_anchor_roi
    if (
        len(robot_roi) != 4
        or not all(0.0 <= float(value) <= 1.0 for value in robot_roi)
        or float(robot_roi[0]) >= float(robot_roi[2])
        or float(robot_roi[1]) >= float(robot_roi[3])
    ):
        raise ValueError(
            "segmentation.robot_anchor_roi must be normalized [x_min, y_min, x_max, y_max]"
        )
    for name in ("robot_green_hsv_lower", "robot_green_hsv_upper"):
        value = getattr(config.segmentation, name)
        if len(value) != 3 or not all(0 <= int(channel) <= 255 for channel in value):
            raise ValueError(f"segmentation.{name} must contain three values in [0, 255]")
    if any(
        int(getattr(config.segmentation, name)) < 0
        for name in (
            "robot_mask_dilation_px",
            "robot_tip_extension_px",
            "robot_mask_hold_frames",
            "robot_mask_temporal_frames",
            "robot_component_link_px",
            "robot_min_residual_pixels",
            "robot_center_hold_frames",
        )
    ):
        raise ValueError("segmentation robot self-filter sizes cannot be negative")
    if config.segmentation.robot_mask_temporal_frames < 1:
        raise ValueError("segmentation.robot_mask_temporal_frames must be positive")
    if not 0.0 <= config.segmentation.robot_reject_overlap <= 1.0:
        raise ValueError("segmentation.robot_reject_overlap must be in [0, 1]")
    if not 0.0 <= config.segmentation.robot_reject_min_overlap <= 1.0:
        raise ValueError("segmentation.robot_reject_min_overlap must be in [0, 1]")
    if not 0.0 < config.segmentation.robot_center_ema_alpha <= 1.0:
        raise ValueError("segmentation.robot_center_ema_alpha must be in (0, 1]")
    if config.reconstruction.max_points < 100:
        raise ValueError("reconstruction.max_points must be at least 100")
    if config.reconstruction.display_confidence_threshold < 0:
        raise ValueError("reconstruction.display_confidence_threshold cannot be negative")
    if config.reconstruction.anchor_interval < 0:
        raise ValueError("reconstruction.anchor_interval cannot be negative")
    if config.reconstruction.max_relative_depth <= 0:
        raise ValueError("reconstruction.max_relative_depth must be positive")
    if (
        config.reconstruction.max_metric_depth_m is not None
        and config.reconstruction.max_metric_depth_m <= 0
    ):
        raise ValueError("reconstruction.max_metric_depth_m must be positive")
    for name in ("focal_length_x", "focal_length_y"):
        value = getattr(config.reconstruction, name)
        if value is not None and value <= 0:
            raise ValueError(f"reconstruction.{name} must be positive")
    reference_depth = config.reconstruction.metric_reference_depth_m
    if reference_depth is not None and reference_depth <= 0:
        raise ValueError("reconstruction.metric_reference_depth_m must be positive")
    roi = config.reconstruction.metric_reference_roi
    if (
        len(roi) != 4
        or not all(0.0 <= float(value) <= 1.0 for value in roi)
        or float(roi[0]) >= float(roi[2])
        or float(roi[1]) >= float(roi[3])
    ):
        raise ValueError(
            "reconstruction.metric_reference_roi must be normalized [x_min, y_min, x_max, y_max]"
        )
    if not 0.0 < config.reconstruction.metric_reference_percentile < 100.0:
        raise ValueError("reconstruction.metric_reference_percentile must be between 0 and 100")
    if config.reconstruction.metric_reference_warmup_frames < 1:
        raise ValueError("reconstruction.metric_reference_warmup_frames must be positive")
    if not 0.0 <= config.reconstruction.metric_reference_ema_alpha <= 1.0:
        raise ValueError("reconstruction.metric_reference_ema_alpha must be in [0, 1]")
    if config.tracking.visual_hold_updates < 0:
        raise ValueError("tracking.visual_hold_updates cannot be negative")
    if config.tracking.obstacle_cloud_hold_updates < 0:
        raise ValueError("tracking.obstacle_cloud_hold_updates cannot be negative")
    if config.tracking.obstacle_cloud_hold_updates > config.tracking.max_missing:
        raise ValueError(
            "tracking.obstacle_cloud_hold_updates cannot exceed tracking.max_missing"
        )
    if config.tracking.confirmation_hits < 1:
        raise ValueError("tracking.confirmation_hits must be positive")
    if not 0.0 < config.tracking.bbox_smoothing_alpha <= 1.0:
        raise ValueError("tracking.bbox_smoothing_alpha must be in (0, 1]")
    if config.tracking.obstacle_center_max_step_m <= 0:
        raise ValueError("tracking.obstacle_center_max_step_m must be positive")
    if config.gui.history_frames < 1 or config.gui.history_stride < 1:
        raise ValueError("GUI history settings must be positive")
    if config.gui.video_fps <= 0:
        raise ValueError("gui.video_fps must be positive")
    if config.scale_mode not in {"relative", "calibrated", "rgbd"}:
        raise ValueError(f"Unsupported scale mode: {config.scale_mode}")
    if config.scale_mode == "calibrated":
        if config.manual_scale is not None and config.manual_scale <= 0:
            raise ValueError("manual_scale must be positive")
        if config.manual_scale is None and reference_depth is None:
            raise ValueError(
                "calibrated scale mode requires manual_scale or reconstruction.metric_reference_depth_m"
            )
