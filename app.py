from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

from realtime_safety.config import load_config
from realtime_safety.export.pointcloud_export import export_ply
from realtime_safety.export.session_logger import SessionLogger
from realtime_safety.export.video_recorder import VideoRecorder
from realtime_safety.gui.dashboard import Dashboard
from realtime_safety.gui.reconstruction_scene import ReconstructionScene3D
from realtime_safety.gui.scene_3d import Scene3D
from realtime_safety.pipeline.video_source import CameraDetectionError
from realtime_safety.scheduler import RealtimePipeline
from realtime_safety.utils.validation import validate_config


def _edge_control_initial_mode(obstacle_backend: str) -> str:
    """Map the three-way obstacle pipeline to Edge's two-way service mode."""

    mode = str(obstacle_backend).strip().lower()
    if mode == "edgetam":
        return "edgetam"
    if mode in {"pointcloud", "yolo"}:
        # YOLO owns the mux output, so the independent Edge process should run
        # without neural refinement until an Edge mode is requested.
        return "pointcloud"
    raise ValueError(f"Unsupported obstacle backend: {obstacle_backend}")


def _simulator_rgbd_projection_parameters(
    simulator_configs: dict,
) -> tuple[object, object, object | None, object | None, float]:
    """Return configured optical extrinsics, crop and depth noise.

    The live ``/sim/camera/pose`` sample supersedes this initial transform.
    Keeping the deterministic YAML fallback makes startup and unit tests
    independent of Gazebo publication order.
    """

    import numpy as np
    from scipy.spatial.transform import Rotation

    camera = simulator_configs["camera"]
    scene = simulator_configs["scene"]
    explicit_pose = camera.get("world_pose")
    if explicit_pose:
        position = np.asarray(explicit_pose["position"], dtype=np.float64)
        world_from_link = Rotation.from_euler(
            "xyz", explicit_pose["rpy_deg"], degrees=True
        ).as_matrix()
    else:
        workspace = np.asarray(
            scene["zones"]["workspace"]["center"], dtype=np.float64
        )
        position = workspace + np.array(
            [
                -float(camera["horizontal_offset_to_workspace_center"]),
                float(camera["lateral_offset"]),
                float(camera["height_above_table"]),
            ]
        )
        target = workspace + np.asarray(
            camera.get("aim_offset", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        forward = target - position
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        world_from_link = np.column_stack((forward, left, up))
    optical_to_link = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    pointcloud = camera.get("pointcloud", {})
    crop_min = pointcloud.get("world_crop_min")
    crop_max = pointcloud.get("world_crop_max")
    noise = camera.get("noise", {})
    depth_noise = (
        float(noise.get("depth_stddev_m", 0.0))
        if bool(noise.get("enabled", False))
        else 0.0
    )
    return (
        position,
        world_from_link @ optical_to_link,
        None if crop_min is None else np.asarray(crop_min, dtype=np.float64),
        None if crop_max is None else np.asarray(crop_max, dtype=np.float64),
        depth_noise,
    )


def _configure_runtime_thread_pools() -> tuple[int, int | None, int | None]:
    """Keep small live frames from fanning out across every CPU core."""

    import cv2

    opencv_threads = max(
        int(os.environ.get("REALTIME_OPENCV_THREADS", "2")), 1
    )
    torch_threads = max(
        int(os.environ.get("REALTIME_TORCH_THREADS", "4")), 1
    )
    cv2.setNumThreads(opencv_threads)
    torch_intra_threads: int | None = None
    torch_interop_threads: int | None = None
    try:
        import torch

        torch.set_num_threads(torch_threads)
        # Inter-op work is limited separately; two schedulers are sufficient
        # for one reconstruction worker and one optional segmentation worker.
        torch.set_num_interop_threads(min(torch_threads, 2))
        torch_intra_threads = torch.get_num_threads()
        torch_interop_threads = torch.get_num_interop_threads()
    except (ImportError, RuntimeError):
        # CPU-only installations and repeated embedded invocations can omit or
        # have already initialized the PyTorch inter-op pool.
        pass
    return cv2.getNumThreads(), torch_intra_threads, torch_interop_threads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive temporal-depth 4D reconstruction viewer and safety dashboard")
    parser.add_argument(
        "--source",
        help="Video path, webcam index, HTTP/RTSP URL, or ros2:///absolute/image_topic",
    )
    parser.add_argument(
        "--camera-qos",
        choices=("sensor_data", "best_effort", "reliable"),
        default=os.environ.get("REALTIME_CAMERA_QOS", "sensor_data"),
        help="ROS 2 image subscription QoS (Isaac Sim default: sensor_data)",
    )
    parser.add_argument(
        "--auto-webcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically detect and start the first readable webcam when --source is omitted",
    )
    parser.add_argument("--profile", default="st4rtrack_viewer", help="Profile name or YAML path")
    parser.add_argument("--mode", choices=("reconstruction", "safety"), help="Override the profile application mode")
    parser.add_argument(
        "--people",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable YOLO person masks, 3D person boxes, centers, and direction arrows",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument(
        "--depth-mode",
        choices=(
            "video_depth",
            "mast3r_slam",
            "st4rtrack",
            "hybrid",
            "fast_depth",
            "rgbd",
        ),
    )
    parser.add_argument("--scale-mode", choices=("relative", "calibrated", "rgbd"))
    parser.add_argument("--manual-scale", type=float)
    parser.add_argument("--focal-x", type=float, help="Webcam intrinsic fx in pixels for 3D reprojection")
    parser.add_argument("--focal-y", type=float, help="Webcam intrinsic fy in pixels for 3D reprojection")
    parser.add_argument("--principal-x", type=float, help="Webcam intrinsic cx in pixels")
    parser.add_argument("--principal-y", type=float, help="Webcam intrinsic cy in pixels")
    parser.add_argument(
        "--depth-reference-m",
        type=float,
        help="Known camera-to-reference distance in metres for live metric scale calibration",
    )
    parser.add_argument(
        "--depth-reference-roi",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Normalized fixed-reference ROI used with --depth-reference-m",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--presentation-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the clean, recording-friendly GUI layout",
    )
    parser.add_argument("--pointcloud-topic", help="Publish PointCloud2 on this absolute ROS 2 topic")
    parser.add_argument(
        "--rgbd-pointcloud-topic",
        help=(
            "Legacy: consume a metric ROS optical-frame PointCloud2. "
            "Prefer --rgbd-depth-topic for image-driven backprojection."
        ),
    )
    parser.add_argument(
        "--rgbd-color-topic",
        default="/rgbd/color/image_raw",
        help="RGB image paired with --rgbd-depth-topic",
    )
    parser.add_argument(
        "--rgbd-depth-topic",
        help="Aligned metric depth Image reconstructed inside 3D Safety",
    )
    parser.add_argument(
        "--rgbd-camera-info-input-topic",
        default="/rgbd/color/camera_info",
        help="CameraInfo paired with the aligned RGB-D images",
    )
    parser.add_argument(
        "--rgbd-camera-pose-topic",
        default="/sim/camera/pose",
        help="Optional world-frame camera link pose used only as RGB-D extrinsics",
    )
    parser.add_argument(
        "--rgbd-generated-world-pointcloud-topic",
        default="/realtime_safety/environment_cloud_world",
        help="Publish the 3D-Safety-generated current RGB-D cloud in world frame",
    )
    parser.add_argument(
        "--rgbd-world-pointcloud-topic",
        default="/rgbd/points_world",
        help="World-frame simulator PointCloud2 used by the geometry debug layer",
    )
    parser.add_argument(
        "--sim-config-root",
        help="Directory containing scene.yaml, camera.yaml and openarm.yaml",
    )
    parser.add_argument("--pointcloud-frame-id", default="realtime_safety_frame")
    parser.add_argument("--pointcloud-rate", type=float, help="Maximum ROS 2 point-cloud publication rate")
    parser.add_argument(
        "--pointcloud-coordinate-mode",
        choices=("internal_z_up", "camera_y_forward", "ros_optical"),
        default="internal_z_up",
        help=(
            "ROS PointCloud2 axes: internal_z_up is x-right/y-forward/z-up; "
            "camera_y_forward is Koch VAMP x-right/y-forward/z-down; "
            "ros_optical is REP-103 x-right/y-down/z-forward"
        ),
    )
    parser.add_argument(
        "--yolo-obstacle-pointcloud-topic",
        help="Publish points inside YOLO obstacle masks on this ROS 2 PointCloud2 topic",
    )
    parser.add_argument(
        "--yolo-obstacle-pointcloud-rate",
        type=float,
        help="Maximum YOLO obstacle PointCloud2 publication rate",
    )
    parser.add_argument(
        "--arm-obstacle-relationship-topic",
        help=(
            "Publish the estimated arm center, obstacle centers, and metric "
            "distances as versioned std_msgs/String JSON"
        ),
    )
    parser.add_argument(
        "--arm-obstacle-relationship-rate",
        type=float,
        help="Maximum arm-obstacle relationship publication rate",
    )
    parser.add_argument("--camera-preview-topic", help="Republish decoded camera frames on this ROS 2 Image topic")
    parser.add_argument("--camera-preview-rate", type=float, default=10.0)
    parser.add_argument(
        "--camera-preview-frame-id",
        default="realtime_safety_frame",
        help="Frame used by the paired RGB Image and CameraInfo topics",
    )
    parser.add_argument(
        "--camera-info-topic",
        help="Publish CameraInfo paired with --camera-preview-topic",
    )
    parser.add_argument("--ros-domain-id", type=int, help="ROS_DOMAIN_ID used by the PointCloud2 publisher")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Do not launch Viser")
    parser.add_argument("--max-frames", type=int, help="Stop capture after this many frames")
    parser.add_argument("--duration", type=float, help="Stop after wall-clock seconds")
    parser.add_argument("--exit-on-end", action="store_true", help="Exit GUI when finite source ends")
    parser.add_argument("--output-dir", help="Session output directory (default: sessions/<timestamp>)")
    parser.add_argument("--no-log", action="store_true", help="Disable streaming JSONL/trajectory CSV")
    parser.add_argument("--record", action="store_true", help="Record annotated MP4")
    parser.add_argument("--export-ply", action="store_true", help="Export the final visible point cloud on exit")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ["REALTIME_CAMERA_QOS"] = args.camera_qos
    opencv_threads, torch_threads, torch_interop_threads = (
        _configure_runtime_thread_pools()
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info(
        "Runtime CPU pools: OpenCV=%d PyTorch=%s interop=%s "
        "OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s",
        opencv_threads,
        torch_threads if torch_threads is not None else "unavailable",
        torch_interop_threads if torch_interop_threads is not None else "unavailable",
        os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        os.environ.get("OMP_NUM_THREADS", "unset"),
    )
    config = load_config(args.profile)
    if args.mode:
        config.mode = args.mode
    if args.people is not None:
        config.people_overlay = args.people
    if args.device:
        config.device = args.device
    if args.depth_mode:
        config.reconstruction.depth_mode = args.depth_mode
    if args.scale_mode:
        config.scale_mode = args.scale_mode
    if args.manual_scale is not None:
        if args.manual_scale <= 0:
            raise ValueError("--manual-scale must be positive")
        config.manual_scale = args.manual_scale
        config.scale_mode = "calibrated"
    if args.focal_x is not None:
        config.reconstruction.focal_length_x = args.focal_x
    if args.focal_y is not None:
        config.reconstruction.focal_length_y = args.focal_y
    if args.principal_x is not None:
        config.reconstruction.principal_point_x = args.principal_x
    if args.principal_y is not None:
        config.reconstruction.principal_point_y = args.principal_y
    if args.depth_reference_m is not None:
        config.reconstruction.metric_reference_depth_m = args.depth_reference_m
        config.scale_mode = "calibrated"
    if args.depth_reference_roi is not None:
        config.reconstruction.metric_reference_roi = tuple(args.depth_reference_roi)
    if args.host:
        config.gui.host = args.host
    if args.port:
        config.gui.port = args.port
    if args.presentation_mode is not None:
        config.gui.presentation_mode = args.presentation_mode
    if args.loop:
        config.video.loop = True
    if args.ros_domain_id is not None:
        if args.ros_domain_id < 0:
            raise ValueError("--ros-domain-id cannot be negative")
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    simulator_configs = None
    if args.rgbd_pointcloud_topic or args.rgbd_depth_topic:
        if not args.sim_config_root:
            raise ValueError(
                "--sim-config-root is required with RGB-D simulator input"
            )
        import yaml

        config_root = Path(args.sim_config_root).expanduser().resolve()
        scene_document = yaml.safe_load((config_root / "scene.yaml").read_text())
        camera_document = yaml.safe_load((config_root / "camera.yaml").read_text())
        openarm_document = yaml.safe_load((config_root / "openarm.yaml").read_text())
        hand_document = yaml.safe_load(
            (config_root / "hand_scenarios.yaml").read_text()
        )
        simulator_configs = {
            "scene": scene_document,
            "camera": camera_document["camera"],
            "openarm": openarm_document["robot"],
            "hand": hand_document,
        }
        config.reconstruction.depth_mode = "rgbd"
        config.reconstruction.depth_mode_options = ["rgbd"]
        config.reconstruction.apriltag_enabled = False
        config.scale_mode = "rgbd"
        config.people_overlay = False
        config.gui.metric_bev_enabled = False
        # The neural model creates the semantic seed. The independent OpenArm
        # resampler back-projects the selected raw model cloud against every
        # fresh depth frame. Keep EdgeTAM's mux input on its raw topic; routing
        # the resampler output back into its own input forms a feedback loop.
        robot = simulator_configs["openarm"]
        config.openarm.base_anchor = "fixed"
        config.openarm.base_position_xyz = tuple(robot["base_position"])
        config.openarm.base_rpy_deg = tuple(robot["base_orientation_rpy_deg"])
        config.openarm.camera_height_m = float(
            simulator_configs["camera"]["height_above_table"]
        )
        config.openarm.camera_downward_angle_deg = float(
            simulator_configs["camera"]["tilt_from_vertical_deg"]
        )
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    validate_config(config)

    pipeline_holder: dict[str, RealtimePipeline] = {}
    dashboard = None
    scene = None
    if not args.headless:
        dashboard = Dashboard(
            config.gui,
            lambda command, value: pipeline_holder["pipeline"].handle_command(command, value),
            reconstruction_only=config.mode == "reconstruction",
            people_overlay=config.people_overlay,
            projection_config=config.reconstruction,
            obstacle_model=config.segmentation.model,
            obstacle_model_options=config.segmentation.model_options,
            obstacle_backend=(
                config.obstacle_perception.backend
                if config.obstacle_perception.enabled
                else None
            ),
            obstacle_backend_options=(
                config.obstacle_perception.backend_options
                if config.obstacle_perception.enabled
                else ()
            ),
            reconstruction_method=config.reconstruction.depth_mode,
            reconstruction_method_options=(
                config.reconstruction.depth_mode_options
            ),
            openarm_control=config.openarm.enabled,
            openarm_hand_config=(
                simulator_configs["hand"]
                if simulator_configs is not None
                else None
            ),
        )
        scene = (
            ReconstructionScene3D(
                dashboard.server, config.gui, config.openarm
            )
            if config.mode == "reconstruction"
            else Scene3D(dashboard.server, config.gui, config.openarm)
        )
        if simulator_configs is not None and isinstance(scene, ReconstructionScene3D):
            scene.configure_simulator_debug(
                simulator_configs["scene"], simulator_configs["camera"]
            )
    output_dir = Path(args.output_dir or f"sessions/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    session_logger = None if args.no_log else SessionLogger(output_dir)
    recorder = VideoRecorder(output_dir / "annotated.mp4") if args.record else None
    pointcloud_publisher = None
    if args.pointcloud_topic:
        from realtime_safety.ros2_bridge.pointcloud_publisher import PointCloudTopicPublisher

        pointcloud_publisher = PointCloudTopicPublisher(
            args.pointcloud_topic,
            args.pointcloud_frame_id,
            max_rate_hz=args.pointcloud_rate,
            coordinate_mode=args.pointcloud_coordinate_mode,
            preserve_source_timestamp=bool(args.rgbd_depth_topic),
        )
    generated_world_pointcloud_publisher = None
    if args.rgbd_depth_topic and args.rgbd_generated_world_pointcloud_topic:
        from realtime_safety.ros2_bridge.pointcloud_publisher import (
            PointCloudTopicPublisher,
        )

        generated_world_pointcloud_publisher = PointCloudTopicPublisher(
            args.rgbd_generated_world_pointcloud_topic,
            "world",
            node_name="realtime_safety_rgbd_world_pointcloud_publisher",
            max_rate_hz=args.pointcloud_rate,
            publish_empty=True,
            coordinate_mode="internal_z_up",
            preserve_source_timestamp=True,
        )
    yolo_obstacle_pointcloud_publisher = None
    yolo_candidate_topic = args.yolo_obstacle_pointcloud_topic
    if (
        not yolo_candidate_topic
        and config.obstacle_perception.enabled
        and "yolo" in config.obstacle_perception.backend_options
    ):
        yolo_candidate_topic = (
            config.obstacle_perception.yolo_obstacle_cloud_topic
        )
    if yolo_candidate_topic:
        from realtime_safety.ros2_bridge.pointcloud_publisher import PointCloudTopicPublisher

        yolo_obstacle_pointcloud_publisher = PointCloudTopicPublisher(
            yolo_candidate_topic,
            args.pointcloud_frame_id,
            node_name="realtime_safety_yolo_obstacle_pointcloud_publisher",
            max_rate_hz=args.yolo_obstacle_pointcloud_rate,
            publish_empty=True,
            coordinate_mode=args.pointcloud_coordinate_mode,
            preserve_source_timestamp=bool(args.rgbd_depth_topic),
        )
    arm_obstacle_relationship_publisher = None
    if args.arm_obstacle_relationship_topic:
        from realtime_safety.ros2_bridge.relationship_publisher import (
            ArmObstacleRelationshipPublisher,
        )

        raw_relationship_publisher = ArmObstacleRelationshipPublisher(
            args.arm_obstacle_relationship_topic,
            args.pointcloud_frame_id,
            max_rate_hz=args.arm_obstacle_relationship_rate,
            coordinate_mode=args.pointcloud_coordinate_mode,
        )
        if config.obstacle_perception.enabled:
            from realtime_safety.ros2_bridge.edge_relationship import (
                EdgeTAMRelationshipBridge,
            )

            arm_obstacle_relationship_publisher = EdgeTAMRelationshipBridge(
                raw_relationship_publisher,
                topic=(
                    config.obstacle_perception.tracked_obstacles_topic
                ),
                source_coordinate_mode=args.pointcloud_coordinate_mode,
                # Until a trained semantic hand checkpoint is installed, do
                # not mislabel arbitrary geometry clusters as human hands.
                class_name="hand_candidate",
                initial_mode=config.obstacle_perception.backend,
                dynamic_enter_speed=config.tracking.dynamic_enter_speed,
                dynamic_exit_speed=config.tracking.dynamic_exit_speed,
                minimum_dynamic_hits=config.tracking.minimum_dynamic_hits,
                manage_publisher_lifecycle=True,
                robot_arm_provider=(
                    getattr(scene, "openarm_robot_state", None)
                    if config.openarm.enabled and scene is not None
                    else None
                ),
            )
        else:
            arm_obstacle_relationship_publisher = raw_relationship_publisher
    camera_preview_publisher = None
    if args.camera_preview_topic:
        from realtime_safety.ros2_bridge.image_publisher import ImageTopicPublisher

        camera_preview_publisher = ImageTopicPublisher(
            args.camera_preview_topic,
            frame_id=args.camera_preview_frame_id,
            max_rate_hz=args.camera_preview_rate,
            camera_info_topic=args.camera_info_topic,
            focal_length_x=config.reconstruction.focal_length_x,
            focal_length_y=config.reconstruction.focal_length_y,
            principal_point_x=config.reconstruction.principal_point_x,
            principal_point_y=config.reconstruction.principal_point_y,
        )
    obstacle_backend_controller = None
    obstacle_cloud_mux = None
    openarm_joint_bridge = None
    openarm_control_bridge = None
    if scene is not None and config.openarm.enabled:
        from realtime_safety.ros2_bridge.openarm_joint_state import (
            OpenArmJointStateBridge,
        )

        update_openarm = getattr(scene, "update_openarm_joint_state", None)
        if callable(update_openarm):
            openarm_joint_bridge = OpenArmJointStateBridge(
                config.openarm.joint_states_topic,
                update_openarm,
            )
        from realtime_safety.ros2_bridge.openarm_control import OpenArmControlBridge

        openarm_control_bridge = OpenArmControlBridge(
            on_status=(
                dashboard.update_openarm_control_status
                if dashboard is not None
                else None
            )
        )
    if config.obstacle_perception.enabled:
        from realtime_safety.ros2_bridge.edgetam_control import (
            EdgeTAMControlBridge,
            EdgeTAMControlStatus,
        )

        def update_edge_status(status: EdgeTAMControlStatus) -> None:
            if dashboard is None:
                return
            pipeline = pipeline_holder.get("pipeline")
            if (
                pipeline is not None
                and pipeline.obstacle_backend_mode == "yolo"
            ):
                # Edge diagnostics continue in the background, but must not
                # move the effective GUI selection away from the YOLO mux.
                return
            diagnostics = dict(status.diagnostics)
            metrics = {
                "edge_status": diagnostics.get(
                    "state",
                    diagnostics.get("pipeline.edge_status", "unknown"),
                ),
                "edge_error": diagnostics.get(
                    "error",
                    diagnostics.get("pipeline.edge_error", ""),
                ),
                "fps": diagnostics.get(
                    "pipeline.fps", diagnostics.get("fps", "--")
                ),
                "edge_latency_ms": diagnostics.get(
                    "latency_ms",
                    diagnostics.get("pipeline.edge_latency_ms", "--"),
                ),
                "track_count": diagnostics.get(
                    "pipeline.track_count", diagnostics.get("track_count", "--")
                ),
                "edge_refined_corrections": diagnostics.get(
                    "refined_corrections",
                    diagnostics.get("pipeline.edge_refined_corrections", "0"),
                ),
                "prompt_count": diagnostics.get(
                    "pipeline.prompt_count", diagnostics.get("prompt_count", "0")
                ),
                "mask_good_count": diagnostics.get(
                    "mask_good_count",
                    diagnostics.get("pipeline.mask_good_count", "0"),
                ),
                "mask_degraded_count": diagnostics.get(
                    "mask_degraded_count",
                    diagnostics.get("pipeline.mask_degraded_count", "0"),
                ),
                "mask_invalid_count": diagnostics.get(
                    "mask_invalid_count",
                    diagnostics.get("pipeline.mask_invalid_count", "0"),
                ),
                "mask_reject_reasons": diagnostics.get(
                    "mask_reject_reasons",
                    diagnostics.get("pipeline.mask_reject_reasons", ""),
                ),
                "background_state": diagnostics.get(
                    "pipeline.background_state",
                    diagnostics.get("background_state", "disabled"),
                ),
                "background_warmup": diagnostics.get(
                    "pipeline.background_warmup", "0/0"
                ),
                "background_calibration": diagnostics.get(
                    "pipeline.background_calibration", "0/0"
                ),
                "background_removed": diagnostics.get(
                    "pipeline.background_removed", "0"
                ),
                "background_baseline_points": diagnostics.get(
                    "pipeline.background_baseline_points", "0"
                ),
                "background_matched_points": diagnostics.get(
                    "pipeline.background_matched_points", "0"
                ),
                "background_alignment_points": diagnostics.get(
                    "pipeline.background_alignment_points", "0"
                ),
                "background_depth_scale": diagnostics.get(
                    "pipeline.background_depth_scale", "1.00000"
                ),
                "background_candidate_depth_scale": diagnostics.get(
                    "pipeline.background_candidate_depth_scale", "1.00000"
                ),
                "background_candidate_alignment_points": diagnostics.get(
                    "pipeline.background_candidate_alignment_points", "0"
                ),
                "background_candidate_alignment_reason": diagnostics.get(
                    "pipeline.background_candidate_alignment_reason",
                    "not_evaluated",
                ),
                "background_alignment_valid": diagnostics.get(
                    "pipeline.background_alignment_valid",
                    diagnostics.get("background_alignment_valid", "false"),
                ),
                "hand_candidate_count": diagnostics.get(
                    "pipeline.hand_candidate_count",
                    diagnostics.get("hand_candidate_count", "0"),
                ),
                "geometry_fallback_track_count": diagnostics.get(
                    "pipeline.geometry_fallback_track_count", "0"
                ),
                "hand_semantic_status": diagnostics.get(
                    "pipeline.hand_semantic_status",
                    diagnostics.get("hand_semantic_status", "disabled"),
                ),
                "hand_rgb_detection_count": diagnostics.get(
                    "pipeline.hand_rgb_detection_count",
                    diagnostics.get("hand_rgb_detection_count", "0"),
                ),
                "hand_semantic_reject_reasons": diagnostics.get(
                    "pipeline.hand_semantic_reject_reasons",
                    diagnostics.get("hand_semantic_reject_reasons", ""),
                ),
                "safety_output_state": diagnostics.get(
                    "pipeline.safety_output_state",
                    diagnostics.get("safety_output_state", "publishing_verified"),
                ),
                "pipeline_level": diagnostics.get(
                    "pipeline.level", "0"
                ),
                "pipeline_message": diagnostics.get(
                    "pipeline.message", ""
                ),
            }
            shown_backend = (
                status.requested_mode
                if status.state == "loading"
                else status.active_mode
            )
            dashboard.update_obstacle_backend_status(
                shown_backend,
                state=status.state,
                detail=status.message,
                metrics=metrics if status.diagnostics else None,
            )

        obstacle_backend_controller = EdgeTAMControlBridge(
            update_edge_status,
            on_debug_image=(
                None
                if dashboard is None
                else dashboard.update_edgetam_debug_image
            ),
            on_obstacle_cloud=(
                scene.update_edge_obstacle_cloud
                if scene is not None
                and callable(
                    getattr(scene, "update_edge_obstacle_cloud", None)
                )
                else None
            ),
            diagnostics_topic=config.obstacle_perception.diagnostics_topic,
            obstacle_cloud_topic=(
                "/edgetam_tracker/obstacle_cloud_realtime"
                if simulator_configs is not None
                else config.obstacle_perception.obstacle_cloud_topic
            ),
            service_name=config.obstacle_perception.control_service,
            initial_mode=_edge_control_initial_mode(
                config.obstacle_perception.backend
            ),
        )
        from realtime_safety.ros2_bridge.obstacle_cloud_mux import (
            ObstacleCloudMux,
        )

        def update_mux_status(status) -> None:
            pipeline = pipeline_holder.get("pipeline")
            if pipeline is not None:
                pipeline.update_obstacle_mux_status(status)

        obstacle_cloud_mux = ObstacleCloudMux(
            edge_topic=(
                config.obstacle_perception.edgetam_obstacle_cloud_topic
            ),
            yolo_topic=config.obstacle_perception.yolo_obstacle_cloud_topic,
            output_topic=config.obstacle_perception.obstacle_cloud_topic,
            initial_mode=config.obstacle_perception.backend,
            on_status=update_mux_status,
        )
    pipeline = RealtimePipeline(
        config,
        dashboard=dashboard,
        scene=scene,
        session_logger=session_logger,
        video_recorder=recorder,
        pointcloud_publisher=pointcloud_publisher,
        yolo_obstacle_pointcloud_publisher=yolo_obstacle_pointcloud_publisher,
        arm_obstacle_relationship_publisher=arm_obstacle_relationship_publisher,
        camera_preview_publisher=camera_preview_publisher,
        obstacle_backend_controller=obstacle_backend_controller,
        obstacle_cloud_mux=obstacle_cloud_mux,
        openarm_joint_bridge=openarm_joint_bridge,
        openarm_control_bridge=openarm_control_bridge,
    )
    pipeline_holder["pipeline"] = pipeline
    rgbd_scene_bridge = None
    simulator_pose_bridge = None
    if args.rgbd_depth_topic:
        from realtime_safety.ros2_bridge.rgbd_scene_bridge import (
            RgbdImageSceneBridge,
            RgbdProjectionConfig,
        )

        assert simulator_configs is not None
        (
            camera_position,
            world_from_optical,
            world_crop_min,
            world_crop_max,
            depth_noise_stddev_m,
        ) = _simulator_rgbd_projection_parameters(simulator_configs)
        camera_config = simulator_configs["camera"]
        scene_config = simulator_configs["scene"]
        rgbd_scene_bridge = RgbdImageSceneBridge(
            args.rgbd_color_topic,
            args.rgbd_depth_topic,
            args.rgbd_camera_info_input_topic,
            pipeline.ingest_external_pointcloud,
            (
                scene.update_simulator_debug_cloud
                if scene is not None
                and callable(getattr(scene, "update_simulator_debug_cloud", None))
                else None
            ),
            camera_pose_topic=args.rgbd_camera_pose_topic,
            on_world_cloud=(
                generated_world_pointcloud_publisher.publish
                if generated_world_pointcloud_publisher is not None
                else None
            ),
            projection_config=RgbdProjectionConfig(
                max_points=config.reconstruction.max_points,
                minimum_depth_m=float(camera_config.get("near_clip", 0.05)),
                maximum_depth_m=float(camera_config.get("far_clip", 4.0)),
                sync_slop_sec=0.02,
                depth_noise_stddev_m=depth_noise_stddev_m,
                noise_seed=int(scene_config.get("seed", 0)),
            ),
            camera_position=camera_position,
            world_from_optical=world_from_optical,
            world_crop_min=world_crop_min,
            world_crop_max=world_crop_max,
        )
    elif args.rgbd_pointcloud_topic:
        from realtime_safety.ros2_bridge.rgbd_scene_bridge import RgbdSceneBridge

        rgbd_scene_bridge = RgbdSceneBridge(
            args.rgbd_pointcloud_topic,
            args.rgbd_world_pointcloud_topic,
            pipeline.ingest_external_pointcloud,
            (
                scene.update_simulator_debug_cloud
                if scene is not None
                and callable(getattr(scene, "update_simulator_debug_cloud", None))
                else None
            ),
            max_points=config.reconstruction.max_points,
        )
    if args.rgbd_pointcloud_topic or args.rgbd_depth_topic:
        if scene is not None and callable(
            getattr(scene, "update_simulator_entity_poses", None)
        ):
            from realtime_safety.ros2_bridge.simulator_pose_bridge import (
                SimulatorPoseBridge,
            )

            simulator_pose_bridge = SimulatorPoseBridge(
                "/sim/ground_truth/scene_poses",
                scene.update_simulator_entity_poses,
            )
    shutdown = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    pipeline.start_workers()
    if generated_world_pointcloud_publisher is not None:
        generated_world_pointcloud_publisher.start()
    if rgbd_scene_bridge is not None:
        rgbd_scene_bridge.start()
    if simulator_pose_bridge is not None:
        simulator_pose_bridge.start()
    try:
        if args.source is not None:
            source = int(args.source) if args.source.isdigit() else args.source
            pipeline.start_source(source, max_frames=args.max_frames)
        elif args.auto_webcam:
            try:
                pipeline.start_source("auto", max_frames=args.max_frames)
            except (CameraDetectionError, RuntimeError) as exc:
                if args.headless:
                    raise
                logging.info("No readable webcam detected; waiting for a GUI upload or source selection: %s", exc)
                if dashboard is not None:
                    dashboard.update_camera_status(f"Webcam: **not detected** — {exc}")
        elif args.headless:
            raise ValueError("--source is required in headless mode when --no-auto-webcam is used")

        deadline = time.perf_counter() + args.duration if args.duration else None
        while not shutdown.is_set():
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if args.exit_on_end and pipeline.source_done:
                break
            if args.headless and pipeline.source_done:
                break
            shutdown.wait(0.1)
    finally:
        final_cloud = pipeline.gui_state.read().pointcloud
        if rgbd_scene_bridge is not None:
            rgbd_scene_bridge.close()
        if simulator_pose_bridge is not None:
            simulator_pose_bridge.close()
        if generated_world_pointcloud_publisher is not None:
            generated_world_pointcloud_publisher.close()
        pipeline.close()
        if scene is not None:
            scene.close()
        if dashboard is not None:
            dashboard.close()
        if args.export_ply and final_cloud is not None:
            export_ply(final_cloud, output_dir / "final_pointcloud.ply")
    snapshot = pipeline.gui_state.read()
    perf = snapshot.performance
    logging.info(
        "Final actual rates: input=%.2f display=%.2f segmentation=%.2f 3D=%.2f safety=%.2f FPS, latency avg/p95=%.1f/%.1f ms, dropped=%d",
        perf.input_fps,
        perf.display_fps,
        perf.segmentation_fps,
        perf.reconstruction_fps,
        perf.safety_fps,
        perf.average_latency_ms,
        perf.p95_latency_ms,
        perf.dropped_frames,
    )
    if pipeline.errors:
        logging.warning("Runtime fallbacks/errors: %s", pipeline.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
