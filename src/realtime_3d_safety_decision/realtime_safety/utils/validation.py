from __future__ import annotations

from realtime_safety.config import AppConfig


def validate_config(config: AppConfig) -> None:
    if config.mode not in {"safety", "reconstruction"}:
        raise ValueError(f"Unsupported application mode: {config.mode}")
    if config.video.queue_size < 1:
        raise ValueError("video.queue_size must be positive")
    if config.safety.target_hz <= 0 or config.segmentation.frequency_hz <= 0:
        raise ValueError("update frequencies must be positive")
    model = str(config.segmentation.model).strip()
    model_options = [str(option).strip() for option in config.segmentation.model_options]
    if not model:
        raise ValueError("segmentation.model cannot be empty")
    if any(not option for option in model_options):
        raise ValueError("segmentation.model_options cannot contain empty values")
    if len(model_options) != len(set(model_options)):
        raise ValueError("segmentation.model_options cannot contain duplicates")
    if model_options and model not in model_options:
        raise ValueError(
            "segmentation.model must be included in segmentation.model_options"
        )
    obstacle_backend = str(config.obstacle_perception.backend).strip()
    obstacle_options = [
        str(option).strip()
        for option in config.obstacle_perception.backend_options
    ]
    supported_obstacle_backends = {"edgetam", "pointcloud", "yolo"}
    if obstacle_backend not in supported_obstacle_backends:
        raise ValueError(
            f"Unsupported obstacle_perception.backend: {obstacle_backend}"
        )
    if config.obstacle_perception.enabled and not obstacle_options:
        raise ValueError("obstacle_perception.backend_options cannot be empty")
    if (
        any(option not in supported_obstacle_backends for option in obstacle_options)
        or len(obstacle_options) != len(set(obstacle_options))
    ):
        raise ValueError(
            "obstacle_perception.backend_options must contain unique "
            "edgetam/pointcloud/yolo values"
        )
    if config.obstacle_perception.enabled and obstacle_backend not in obstacle_options:
        raise ValueError(
            "obstacle_perception.backend must be included in backend_options"
        )
    for name in (
        "diagnostics_topic",
        "tracked_obstacles_topic",
        "obstacle_cloud_topic",
        "edgetam_obstacle_cloud_topic",
        "yolo_obstacle_cloud_topic",
        "control_service",
    ):
        topic = str(getattr(config.obstacle_perception, name)).strip()
        if (
            config.obstacle_perception.enabled
            and (
                not topic.startswith("/")
                or any(character.isspace() for character in topic)
            )
        ):
            raise ValueError(
                f"obstacle_perception.{name} must be an absolute ROS name"
            )
    obstacle_topics = {
        str(config.obstacle_perception.obstacle_cloud_topic),
        str(config.obstacle_perception.edgetam_obstacle_cloud_topic),
        str(config.obstacle_perception.yolo_obstacle_cloud_topic),
    }
    if config.obstacle_perception.enabled and len(obstacle_topics) != 3:
        raise ValueError(
            "obstacle_perception candidate/output cloud topics must be distinct"
        )
    if not 0.0 < config.segmentation.tracking_confidence <= config.segmentation.confidence <= 1.0:
        raise ValueError(
            "segmentation confidence must satisfy 0 < tracking_confidence <= confidence <= 1"
        )
    if config.segmentation.hand_maximum_hands < 1:
        raise ValueError("segmentation.hand_maximum_hands must be positive")
    if config.segmentation.hand_model_complexity not in {0, 1}:
        raise ValueError("segmentation.hand_model_complexity must be 0 or 1")
    if config.segmentation.hand_mask_padding_px < 0:
        raise ValueError("segmentation.hand_mask_padding_px cannot be negative")
    if config.segmentation.hand_temporal_hold_frames < 0:
        raise ValueError(
            "segmentation.hand_temporal_hold_frames cannot be negative"
        )
    if config.segmentation.hand_minimum_flow_points < 1:
        raise ValueError(
            "segmentation.hand_minimum_flow_points must be positive"
        )
    if config.segmentation.hand_maximum_flow_error <= 0.0:
        raise ValueError(
            "segmentation.hand_maximum_flow_error must be positive"
        )
    if not 0.0 < config.segmentation.hand_temporal_confidence_decay <= 1.0:
        raise ValueError(
            "segmentation.hand_temporal_confidence_decay must be in (0, 1]"
        )
    for name in (
        "hand_minimum_detection_confidence",
        "hand_minimum_tracking_confidence",
    ):
        value = float(getattr(config.segmentation, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"segmentation.{name} must be in [0, 1]")
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
    fixed_center = config.segmentation.robot_fixed_center_xyz
    if fixed_center is not None and (
        len(fixed_center) != 3
        or not all(float("-inf") < float(value) < float("inf") for value in fixed_center)
    ):
        raise ValueError(
            "segmentation.robot_fixed_center_xyz must contain three finite values"
        )
    fixed_center_xy = config.segmentation.robot_fixed_center_xy
    if (
        len(fixed_center_xy) != 2
        or not all(0.0 <= float(value) <= 1.0 for value in fixed_center_xy)
    ):
        raise ValueError(
            "segmentation.robot_fixed_center_xy must contain normalized x/y"
        )
    if not 0.0 < config.segmentation.robot_fixed_center_confidence <= 1.0:
        raise ValueError(
            "segmentation.robot_fixed_center_confidence must be in (0, 1]"
        )
    if config.segmentation.robot_prefer_fixed_center and fixed_center is None:
        raise ValueError(
            "segmentation.robot_prefer_fixed_center requires robot_fixed_center_xyz"
        )
    if config.reconstruction.max_points < 100:
        raise ValueError("reconstruction.max_points must be at least 100")
    supported_depth_modes = {
        "video_depth",
        "mast3r_slam",
        "st4rtrack",
        "hybrid",
        "fast_depth",
        "rgbd",
    }
    depth_mode = str(config.reconstruction.depth_mode).strip()
    depth_options = [
        str(option).strip()
        for option in config.reconstruction.depth_mode_options
    ]
    if depth_mode not in supported_depth_modes:
        raise ValueError(f"Unsupported reconstruction.depth_mode: {depth_mode}")
    if not depth_options:
        raise ValueError("reconstruction.depth_mode_options cannot be empty")
    if (
        any(option not in supported_depth_modes for option in depth_options)
        or len(depth_options) != len(set(depth_options))
    ):
        raise ValueError(
            "reconstruction.depth_mode_options must contain unique supported methods"
        )
    if depth_mode not in depth_options:
        raise ValueError(
            "reconstruction.depth_mode must be included in depth_mode_options"
        )
    if config.reconstruction.mast3r_slam_image_size not in {224, 512}:
        raise ValueError("reconstruction.mast3r_slam_image_size must be 224 or 512")
    if config.reconstruction.mast3r_slam_confidence_threshold < 0:
        raise ValueError(
            "reconstruction.mast3r_slam_confidence_threshold cannot be negative"
        )
    if config.reconstruction.mast3r_slam_startup_timeout_s <= 0:
        raise ValueError(
            "reconstruction.mast3r_slam_startup_timeout_s must be positive"
        )
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
    if config.reconstruction.apriltag_enabled:
        if config.reconstruction.apriltag_size_m <= 0.0:
            raise ValueError("reconstruction.apriltag_size_m must be positive")
        if config.reconstruction.apriltag_detection_interval < 1:
            raise ValueError("reconstruction.apriltag_detection_interval must be positive")
        if config.reconstruction.apriltag_warmup_detections < 1:
            raise ValueError("reconstruction.apriltag_warmup_detections must be positive")
        if config.reconstruction.apriltag_hold_frames < 0:
            raise ValueError("reconstruction.apriltag_hold_frames cannot be negative")
        if not 0.0 < config.reconstruction.apriltag_scale_ema_alpha <= 1.0:
            raise ValueError("reconstruction.apriltag_scale_ema_alpha must be in (0, 1]")
        if not 0.0 < config.reconstruction.apriltag_max_ratio_spread < 1.0:
            raise ValueError("reconstruction.apriltag_max_ratio_spread must be in (0, 1)")
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
    if not 0.0 <= config.gui.metric_bev_obstacle_height_m <= 1.0:
        raise ValueError(
            "gui.metric_bev_obstacle_height_m must be within [0, 1]"
        )
    if config.openarm.enabled:
        if not config.openarm.joint_states_topic.startswith("/") or any(
            character.isspace()
            for character in config.openarm.joint_states_topic
        ):
            raise ValueError(
                "openarm.joint_states_topic must be an absolute ROS topic"
            )
        if config.openarm.model != "openarm_v1.0_bimanual":
            raise ValueError(f"Unsupported openarm.model: {config.openarm.model}")
        for name in ("base_position_xyz", "base_rpy_deg", "base_from_apriltag_xyz"):
            values = getattr(config.openarm, name)
            if len(values) != 3 or not all(
                float("-inf") < float(value) < float("inf")
                for value in values
            ):
                raise ValueError(f"openarm.{name} must contain three finite values")
        if config.openarm.stale_after_s <= 0.0:
            raise ValueError("openarm.stale_after_s must be positive")
        if config.openarm.base_anchor not in {"fixed", "apriltag"}:
            raise ValueError("openarm.base_anchor must be fixed or apriltag")
        if config.openarm.camera_height_m <= 0.0:
            raise ValueError("openarm.camera_height_m must be positive")
        if not 0.0 <= config.openarm.camera_downward_angle_deg <= 90.0:
            raise ValueError(
                "openarm.camera_downward_angle_deg must be within [0, 90]"
            )
    if config.scale_mode not in {"relative", "calibrated", "rgbd"}:
        raise ValueError(f"Unsupported scale mode: {config.scale_mode}")
    if config.scale_mode == "calibrated":
        if config.manual_scale is not None and config.manual_scale <= 0:
            raise ValueError("manual_scale must be positive")
        if (
            config.manual_scale is None
            and reference_depth is None
            and not config.reconstruction.apriltag_enabled
        ):
            raise ValueError(
                "calibrated scale mode requires manual_scale, metric_reference_depth_m, or AprilTag calibration"
            )
