from __future__ import annotations

"""ROS 2 adapter for the point-cloud-first EdgeTAM obstacle tracker.

The safety-critical geometry path in this module does not depend on EdgeTAM.
RGB segmentation is loaded and executed asynchronously and may only refine a
cluster after the mask passes explicit image/depth/3D consistency gates.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32, Header
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from realtime_3d_safety_decision.msg import (
    TrackedObstacle,
    TrackedObstacleArray,
)
from realtime_safety.edgetam_tracker.cluster_extractor import (
    ClusterExtractor,
    ClusterExtractorConfig,
)
from realtime_safety.edgetam_tracker.edgetam_wrapper import (
    EdgeTAMConfig,
    EdgeTAMError,
    EdgeTAMResult,
    EdgeTAMWrapper,
)
from realtime_safety.edgetam_tracker.mask_pointcloud_fusion import (
    FusionConfig,
    fuse_mask_with_cloud,
)
from realtime_safety.edgetam_tracker.intrusion_gate import (
    SuddenIntrusionGate,
    SuddenIntrusionGateConfig,
)
from realtime_safety.edgetam_tracker.hand_semantic_gate import (
    HandDetection,
    HandSemanticGate,
    HandSemanticGateConfig,
    MediaPipeHandDetector,
    MediaPipeHandDetectorConfig,
)
from realtime_safety.edgetam_tracker.models import (
    AABB,
    CloudFrame,
    Cluster3D,
    MaskObservation,
    MaskQuality,
    PointCloudQuality,
    ProjectionPrompt,
    TrackEstimate,
    TrackingState,
)
from realtime_safety.edgetam_tracker.pointcloud_preprocessor import (
    PointCloudPreprocessor,
    PointCloudPreprocessorConfig,
    StaticBackgroundFilter,
    StaticBackgroundFilterConfig,
    StaticBackgroundState,
)
from realtime_safety.edgetam_tracker.pointcloud_tracker import (
    PointCloudTracker,
    PointCloudTrackerConfig,
)
from realtime_safety.edgetam_tracker.projection_utils import (
    ProjectionConfig,
    project_points,
    project_track,
    transform_points as points_in_camera_frame,
)
from realtime_safety.edgetam_tracker.quality import (
    ConfidenceConfig,
    MaskQualityResult,
    MaskQualityConfig,
    PointCloudQualityConfig,
    RepromptConfig,
    apply_mask_quality,
    apply_pointcloud_quality,
    compute_fused_confidence,
    decide_reprompt,
)
from realtime_safety.edgetam_tracker.robot_self_filter import (
    LinkSphere,
    RobotSelfFilter,
    RobotSelfFilterConfig,
    SelfFilterStatus,
)
from realtime_safety.edgetam_tracker.ros_utils import (
    make_marker_array,
    make_pointcloud2,
    make_rgb_image_message,
    make_tracked_obstacle_array,
    render_debug_image,
    track_color,
)
from realtime_safety.edgetam_tracker.sensor_sync import (
    RosSensorSynchronizer,
    SensorBundle,
    camera_intrinsics_from_info,
    depth_message_to_meters,
    depth_to_cloud,
    image_message_to_array,
    pointcloud2_to_cloud,
    quaternion_matrix_xyzw,
    stamp_to_seconds,
    transform_cloud,
    validate_timestamps,
)


def _fresh_measured_obstacle_cloud(
    tracks: Iterable[TrackEstimate],
    output_stamp: float,
    *,
    timestamp_tolerance_sec: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only points measured in the current geometry frame.

    Track boxes and kinematics intentionally survive short occlusions, but a
    cached ``source_points`` array is not a prediction. Re-publishing those old
    samples made the GUI leave a coloured cloud at the last/edge position after
    the physical obstacle had gone.
    """

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    tolerance = max(float(timestamp_tolerance_sec), 0.0)
    for track in tracks:
        measured_now = (
            track.missed_count == 0
            and track.pointcloud_quality is not PointCloudQuality.INVALID
            and abs(float(track.last_measurement_timestamp) - output_stamp)
            <= tolerance
        )
        if not measured_now or not len(track.source_points):
            continue
        points.append(track.source_points)
        colors.append(
            np.broadcast_to(
                track_color(track.track_id),
                (len(track.source_points), 3),
            ).copy()
        )
    return (
        np.concatenate(points, axis=0)
        if points
        else np.empty((0, 3), dtype=np.float32),
        np.concatenate(colors, axis=0)
        if colors
        else np.empty((0, 3), dtype=np.uint8),
    )


_DEFAULTS: dict[str, Any] = {
    "input_mode": "live",
    "topics.rgb_image": "/realtime_safety/camera/image_raw",
    "topics.depth_image": "",
    "topics.camera_info": "",
    "topics.pointcloud": "/realtime_safety/pointcloud",
    "topics.joint_states": "/joint_states",
    "topics.output_obstacles": "/edgetam_tracker/obstacles",
    "topics.output_obstacle_cloud": "/edgetam_tracker/obstacle_cloud",
    "topics.output_legacy_obstacle_cloud": "/realtime_safety/yolo_obstacles/pointcloud",
    "topics.output_self_filtered_cloud": "/edgetam_tracker/self_filtered_cloud",
    "topics.output_debug_image": "/edgetam_tracker/debug_image",
    "topics.output_debug_cloud": "/edgetam_tracker/debug_cloud",
    "topics.output_markers": "/edgetam_tracker/markers",
    "topics.output_diagnostics": "/edgetam_tracker/diagnostics",
    "topics.output_fps": "/edgetam_tracker/fps",
    "topics.output_latency_ms": "/edgetam_tracker/latency_ms",
    "topics.set_enabled_service": "/edgetam_tracker/set_enabled",
    "frames.tracking_frame": "realtime_safety_frame",
    "frames.robot_base_frame": "realtime_safety_frame",
    "frames.camera_frame": "",
    "frames.tf_timeout_sec": 0.08,
    "sync.queue_size": 10,
    "sync.slop_sec": 0.08,
    "sync.max_data_age_sec": 0.35,
    "sync.allow_static_camera_info": True,
    "sync.pointcloud_fallback_delay_sec": 0.10,
    "sync.sensor_stale_timeout_sec": 0.50,
    "workspace.min_x": -1.5,
    "workspace.max_x": 1.5,
    "workspace.min_y": 0.05,
    "workspace.max_y": 4.0,
    "workspace.min_z": -2.0,
    "workspace.max_z": 2.0,
    "pointcloud.voxel_size": 0.025,
    "pointcloud.remove_outliers": True,
    "pointcloud.outlier_method": "statistical",
    "pointcloud.outlier_mean_k": 16,
    "pointcloud.outlier_stddev": 2.0,
    "pointcloud.outlier_radius_m": 0.08,
    "pointcloud.outlier_min_neighbors": 3,
    "pointcloud.remove_plane": False,
    "pointcloud.plane_distance_threshold": 0.015,
    "pointcloud.plane_ransac_iterations": 80,
    "pointcloud.plane_min_inliers": 30,
    "pointcloud.plane_min_inlier_ratio": 0.10,
    "pointcloud.plane_normal_axis": [],
    "pointcloud.plane_max_angle_deg": 30.0,
    "pointcloud.plane_min_distance_m": 0.0,
    "pointcloud.plane_max_distance_m": 1000000.0,
    "pointcloud.minimum_depth_m": 0.05,
    "pointcloud.maximum_depth_m": 4.0,
    "pointcloud.use_high_resolution_for_final_geometry": True,
    "background.enabled": False,
    "background.warmup_frames": 0,
    "background.calibration_frames": 12,
    "background.voxel_size": 0.015,
    "background.distance_threshold_m": 0.03,
    "background.minimum_baseline_points": 100,
    "background.ray_depth_enabled": False,
    "background.depth_axis": 1,
    "background.horizontal_axis": 0,
    "background.vertical_axis": 2,
    "background.ray_distance_threshold": 0.025,
    "background.alignment_enabled": True,
    "background.alignment_min_points": 80,
    "background.alignment_ratio_tolerance": 0.04,
    "background.maximum_scale_change": 0.35,
    "background.alignment_color_distance": 55.0,
    "background.alignment_require_color": False,
    "background.alignment_periphery_fraction": 0.18,
    "background.alignment_min_support_ratio": 0.60,
    "background.alignment_min_span_ratio": 0.65,
    "background.alignment_min_points_per_tile": 3,
    "background.alignment_min_occupied_tiles": 6,
    "background.alignment_window_frames": 7,
    "background.alignment_min_valid_frames": 5,
    "background.alignment_max_relative_mad": 0.0075,
    "background.alignment_max_window_range": 0.035,
    "background.alignment_max_step_fraction": 0.03,
    "background.alignment_max_upward_rate_per_sec": 0.01,
    "background.alignment_hold_frames": 2,
    "self_filter.enabled": False,
    "self_filter.mode": "tf_spheres",
    "self_filter.link_frames": [],
    "self_filter.link_radii_m": [],
    "self_filter.padding": 0.03,
    "self_filter.fail_closed": True,
    "clustering.method": "dbscan",
    "clustering.tolerance": 0.08,
    "clustering.min_points": 12,
    "clustering.emergency_min_points": 3,
    "clustering.max_points": 100000,
    "clustering.min_dimension": 0.015,
    "clustering.max_dimension": 2.5,
    "clustering.maximum_clusters": 64,
    "clustering.final_geometry_padding_m": 0.035,
    "clustering.depth_axis": 1,
    "tracking.enabled": True,
    "tracking.confirmation_hits": 3,
    "tracking.maximum_missed_frames": 5,
    "tracking.maximum_association_distance": 0.35,
    "tracking.maximum_mahalanobis_distance": 6.0,
    "tracking.maximum_size_log_difference": 1.6,
    "tracking.maximum_count_log_difference": 2.5,
    "tracking.maximum_association_cost": 0.82,
    "tracking.maximum_occluded_frames": 3,
    "tracking.occluded_retention_sec": 0.5,
    "tracking.lost_retention_sec": 1.0,
    "tracking.maximum_prediction_age_sec": 1.0,
    "tracking.maximum_tentative_misses": 1,
    "tracking.tentative_retention_sec": 0.35,
    "tracking.missed_confidence_decay": 0.82,
    "tracking.measured_velocity_blend": 0.30,
    "tracking.process_noise": 0.5,
    "tracking.measurement_noise": 0.03,
    "tracking.prediction_horizons": [0.2, 0.5, 1.0],
    "tracking.centroid_cost_weight": 0.45,
    "tracking.bbox_cost_weight": 0.20,
    "tracking.size_cost_weight": 0.15,
    "tracking.point_count_cost_weight": 0.10,
    "tracking.mask_iou_cost_weight": 0.10,
    "hand_candidate.enabled": False,
    "hand_candidate.minimum_seed_speed_mps": 0.04,
    "hand_candidate.minimum_motion_hits": 2,
    "hand_candidate.maximum_seed_age_frames": 12,
    "hand_semantics.enabled": False,
    "hand_semantics.maximum_hands": 4,
    "hand_semantics.minimum_detection_confidence": 0.50,
    "hand_semantics.minimum_tracking_confidence": 0.45,
    "hand_semantics.model_complexity": 0,
    "hand_semantics.mask_padding_pixels": 4,
    "hand_semantics.static_image_mode": False,
    "hand_semantics.rotation_augmentation_degrees": [0],
    "hand_semantics.cycle_rotation_augmentation": False,
    "hand_semantics.temporal_hold_frames": 8,
    "hand_semantics.temporal_confidence_decay": 0.94,
    "hand_semantics.minimum_flow_points": 4,
    "hand_semantics.maximum_flow_error": 30.0,
    # MediaPipe's convex hand hull can include table pixels between fingers.
    # Keep the nearest coherent depth layer instead of turning all background
    # rays inside the 2-D hull into one oversized obstacle.
    "hand_semantics.foreground_depth_quantile": 0.10,
    "hand_semantics.foreground_depth_span_m": 0.10,
    "hand_semantics.minimum_box_iou": 0.05,
    "hand_semantics.minimum_projection_coverage": 0.30,
    "hand_semantics.minimum_positive_point_coverage": 0.40,
    "hand_semantics.hold_frames": 8,
    "projection.box_padding_px": 8,
    "projection.minimum_box_width": 6,
    "projection.minimum_box_height": 6,
    "projection.maximum_box_area_ratio": 1.0,
    "projection.dilation_pixels": 3,
    "projection.positive_point_count": 5,
    "projection.negative_point_count": 0,
    "projection.use_projection_mask_prompt": True,
    "projection.minimum_projected_points": 6,
    "projection.camera_axis_mode": "x_right_y_forward_z_down",
    "edgetam.enabled": True,
    "edgetam.repository_path": "third_party/EdgeTAM",
    "edgetam.checkpoint_path": "models/edgetam/edgetam.pt",
    "edgetam.model_config": "configs/edgetam.yaml",
    "edgetam.device": "cuda",
    "edgetam.precision": "auto",
    "edgetam.input_width": 1024,
    "edgetam.input_height": 1024,
    "edgetam.prompt_confirmed_tracks_only": True,
    "edgetam.minimum_prompt_interval_sec": 0.5,
    "edgetam.maximum_objects": 16,
    "edgetam.clear_memory_on_reset": True,
    "edgetam.rolling_window_frames": 2,
    "edgetam.jpeg_quality": 95,
    "edgetam.offload_video_to_cpu": False,
    "edgetam.offload_state_to_cpu": False,
    "fusion.mask_erode_pixels": 1,
    "fusion.mask_dilate_pixels": 0,
    "fusion.minimum_valid_depth_ratio": 0.35,
    "fusion.minimum_mask_cluster_iou": 0.25,
    "fusion.maximum_centroid_difference_m": 0.25,
    "fusion.maximum_mask_area_change_ratio": 2.5,
    "fusion.re_prompt_after_invalid_frames": 2,
    "fusion.depth_gate_m": 0.12,
    "fusion.spatial_gate_padding_m": 0.06,
    "fusion.minimum_fused_points": 10,
    "quality.mask_good_threshold": 0.70,
    "quality.mask_degraded_threshold": 0.40,
    "quality.mask_score_weight": 0.10,
    "quality.projected_cluster_coverage_weight": 0.20,
    "quality.valid_depth_ratio_weight": 0.15,
    "quality.mask_cluster_iou_weight": 0.20,
    "quality.temporal_mask_iou_weight": 0.15,
    "quality.prediction_consistency_weight": 0.15,
    "quality.area_change_weight": 0.05,
    "quality.pointcloud_weight": 0.50,
    "quality.temporal_tracking_weight": 0.30,
    "quality.mask_consistency_weight": 0.20,
    "safety.emergency_distance_m": 0.25,
    "safety.base_uncertainty_margin_m": 0.03,
    "safety.maximum_uncertainty_margin_m": 0.25,
    "safety.tf_failure_policy": "hold",
    "compatibility.publish_legacy_obstacle_alias": False,
    "performance.use_latest_frame_only": True,
    "performance.maximum_queue_length": 1,
    "performance.publish_debug_image": True,
    "performance.publish_debug_cloud": True,
    "performance.publish_markers": True,
    "performance.profile_latency": True,
    "performance.prediction_publish_rate_hz": 10.0,
    "performance.diagnostics_rate_hz": 2.0,
}


def _background_output_is_trusted(
    state: StaticBackgroundState,
    *,
    background_enabled: bool,
    ray_depth_enabled: bool,
    alignment_valid: bool,
) -> bool:
    """Return whether background subtraction may seed hand-only output.

    A disabled background filter leaves geometry unchanged and is therefore a
    valid input to the later RGB hand gate.  An enabled filter must finish its
    baseline first.  Ray-depth subtraction additionally needs a trustworthy
    depth-scale alignment; otherwise the complete fixed scene can look like
    foreground and must never be published as a hand.
    """

    if not background_enabled:
        return True
    if state is not StaticBackgroundState.READY:
        return False
    return not ray_depth_enabled or bool(alignment_valid)


@dataclass(slots=True)
class _EdgeFrameContext:
    """Exact immutable-by-convention geometry/RGB context for one async job."""

    sequence: int
    node_generation: int
    frame_index: int
    geometry_stamp: float
    rgb_stamp: float
    submitted_monotonic: float
    publication_serial: int
    header: Header
    rgb_header: Header
    rgb: np.ndarray
    tracks: tuple[TrackEstimate, ...]
    clusters: tuple[Cluster3D, ...]
    raw_cloud: CloudFrame
    prompts: dict[int, ProjectionPrompt]
    tracking_to_camera: np.ndarray | None
    fusion_config: FusionConfig


class EdgeTAMPointCloudTrackerNode(Node):
    """PointCloud2/depth driven tracker with optional asynchronous EdgeTAM."""

    def __init__(
        self, *, parameter_overrides: list[Any] | None = None
    ) -> None:
        super().__init__(
            "edgetam_pointcloud_tracker",
            parameter_overrides=parameter_overrides,
            automatically_declare_parameters_from_overrides=True,
        )
        for name, default in _DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        self._tracking_frame = str(self._p("frames.tracking_frame"))
        self._base_frame = str(self._p("frames.robot_base_frame"))
        if not self._tracking_frame:
            raise ValueError("frames.tracking_frame must not be empty")
        self._tf_timeout = float(self._p("frames.tf_timeout_sec"))
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        workspace_min = np.array(
            [
                self._p("workspace.min_x"),
                self._p("workspace.min_y"),
                self._p("workspace.min_z"),
            ],
            dtype=np.float32,
        )
        workspace_max = np.array(
            [
                self._p("workspace.max_x"),
                self._p("workspace.max_y"),
                self._p("workspace.max_z"),
            ],
            dtype=np.float32,
        )
        self._preprocessor = PointCloudPreprocessor(
            PointCloudPreprocessorConfig(
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                voxel_size=float(self._p("pointcloud.voxel_size")),
                remove_outliers=bool(self._p("pointcloud.remove_outliers")),
                outlier_method=str(self._p("pointcloud.outlier_method")),
                outlier_mean_k=int(self._p("pointcloud.outlier_mean_k")),
                outlier_stddev=float(self._p("pointcloud.outlier_stddev")),
                outlier_radius=float(self._p("pointcloud.outlier_radius_m")),
                outlier_min_neighbors=int(
                    self._p("pointcloud.outlier_min_neighbors")
                ),
                remove_plane=bool(self._p("pointcloud.remove_plane")),
                plane_distance_threshold=float(
                    self._p("pointcloud.plane_distance_threshold")
                ),
                plane_iterations=int(
                    self._p("pointcloud.plane_ransac_iterations")
                ),
                plane_min_inliers=int(
                    self._p("pointcloud.plane_min_inliers")
                ),
                plane_min_inlier_ratio=float(
                    self._p("pointcloud.plane_min_inlier_ratio")
                ),
                plane_normal_axis=self._plane_normal_axis_parameter(),
                plane_max_angle_deg=float(
                    self._p("pointcloud.plane_max_angle_deg")
                ),
                plane_min_distance=float(
                    self._p("pointcloud.plane_min_distance_m")
                ),
                plane_max_distance=float(
                    self._p("pointcloud.plane_max_distance_m")
                ),
            )
        )
        self._background_filter = StaticBackgroundFilter(
            StaticBackgroundFilterConfig(
                enabled=bool(self._p("background.enabled")),
                warmup_frames=int(self._p("background.warmup_frames")),
                calibration_frames=int(
                    self._p("background.calibration_frames")
                ),
                voxel_size=float(self._p("background.voxel_size")),
                distance_threshold=float(
                    self._p("background.distance_threshold_m")
                ),
                minimum_baseline_points=int(
                    self._p("background.minimum_baseline_points")
                ),
                ray_depth_enabled=bool(
                    self._p("background.ray_depth_enabled")
                ),
                depth_axis=int(self._p("background.depth_axis")),
                horizontal_axis=int(
                    self._p("background.horizontal_axis")
                ),
                vertical_axis=int(self._p("background.vertical_axis")),
                ray_distance_threshold=float(
                    self._p("background.ray_distance_threshold")
                ),
                alignment_enabled=bool(
                    self._p("background.alignment_enabled")
                ),
                alignment_min_points=int(
                    self._p("background.alignment_min_points")
                ),
                alignment_ratio_tolerance=float(
                    self._p("background.alignment_ratio_tolerance")
                ),
                maximum_scale_change=float(
                    self._p("background.maximum_scale_change")
                ),
                alignment_color_distance=float(
                    self._p("background.alignment_color_distance")
                ),
                alignment_require_color=bool(
                    self._p("background.alignment_require_color")
                ),
                alignment_periphery_fraction=float(
                    self._p("background.alignment_periphery_fraction")
                ),
                alignment_min_support_ratio=float(
                    self._p("background.alignment_min_support_ratio")
                ),
                alignment_min_span_ratio=float(
                    self._p("background.alignment_min_span_ratio")
                ),
                alignment_min_points_per_tile=int(
                    self._p("background.alignment_min_points_per_tile")
                ),
                alignment_min_occupied_tiles=int(
                    self._p("background.alignment_min_occupied_tiles")
                ),
                alignment_window_frames=int(
                    self._p("background.alignment_window_frames")
                ),
                alignment_min_valid_frames=int(
                    self._p("background.alignment_min_valid_frames")
                ),
                alignment_max_relative_mad=float(
                    self._p("background.alignment_max_relative_mad")
                ),
                alignment_max_window_range=float(
                    self._p("background.alignment_max_window_range")
                ),
                alignment_max_step_fraction=float(
                    self._p("background.alignment_max_step_fraction")
                ),
                alignment_max_upward_rate_per_sec=float(
                    self._p("background.alignment_max_upward_rate_per_sec")
                ),
                alignment_hold_frames=int(
                    self._p("background.alignment_hold_frames")
                ),
            )
        )
        self._self_filter = RobotSelfFilter(
            RobotSelfFilterConfig(
                enabled=bool(self._p("self_filter.enabled")),
                padding=float(self._p("self_filter.padding")),
                fail_closed=bool(self._p("self_filter.fail_closed")),
            )
        )
        self._clusterer = ClusterExtractor(
            ClusterExtractorConfig(
                method=str(self._p("clustering.method")),
                tolerance=float(self._p("clustering.tolerance")),
                min_points=int(self._p("clustering.min_points")),
                max_points=int(self._p("clustering.max_points")),
                min_dimension=float(self._p("clustering.min_dimension")),
                max_dimension=float(self._p("clustering.max_dimension")),
                depth_axis=int(self._p("clustering.depth_axis")),
                dbscan_min_samples=max(
                    min(int(self._p("clustering.min_points")), 8), 3
                ),
                sparse_point_threshold=max(
                    int(self._p("clustering.min_points")) * 2, 20
                ),
                emergency_distance=float(
                    self._p("safety.emergency_distance_m")
                ),
                emergency_min_points=int(
                    self._p("clustering.emergency_min_points")
                ),
            )
        )

        horizons = tuple(
            sorted(
                {
                    float(value)
                    for value in self._p("tracking.prediction_horizons")
                    if float(value) > 0.0
                }
            )
        )
        size_weight = float(self._p("tracking.bbox_cost_weight")) + float(
            self._p("tracking.size_cost_weight")
        )
        self._tracker_config = PointCloudTrackerConfig(
            confirmation_hits=int(self._p("tracking.confirmation_hits")),
            emergency_confirmation_distance=float(
                self._p("safety.emergency_distance_m")
            ),
            maximum_association_distance=float(
                self._p("tracking.maximum_association_distance")
            ),
            maximum_mahalanobis_distance=float(
                self._p("tracking.maximum_mahalanobis_distance")
            ),
            maximum_size_log_difference=float(
                self._p("tracking.maximum_size_log_difference")
            ),
            maximum_count_log_difference=float(
                self._p("tracking.maximum_count_log_difference")
            ),
            maximum_association_cost=float(
                self._p("tracking.maximum_association_cost")
            ),
            maximum_occluded_frames=int(
                self._p("tracking.maximum_occluded_frames")
            ),
            occluded_retention_seconds=float(
                self._p("tracking.occluded_retention_sec")
            ),
            maximum_missed_frames=int(
                self._p("tracking.maximum_missed_frames")
            ),
            lost_retention_seconds=float(
                self._p("tracking.lost_retention_sec")
            ),
            maximum_prediction_age_seconds=float(
                self._p("tracking.maximum_prediction_age_sec")
            ),
            maximum_tentative_misses=int(
                self._p("tracking.maximum_tentative_misses")
            ),
            tentative_retention_seconds=float(
                self._p("tracking.tentative_retention_sec")
            ),
            missed_confidence_decay=float(
                self._p("tracking.missed_confidence_decay")
            ),
            measured_velocity_blend=float(
                self._p("tracking.measured_velocity_blend")
            ),
            acceleration_process_variance=float(
                self._p("tracking.process_noise")
            ),
            measurement_variance=float(
                self._p("tracking.measurement_noise")
            ),
            prediction_horizon_seconds=max(horizons, default=0.0),
            prediction_step_seconds=min(horizons, default=0.2),
            prediction_horizons_seconds=horizons,
            centroid_cost_weight=float(
                self._p("tracking.centroid_cost_weight")
            ),
            aabb_size_cost_weight=size_weight,
            point_count_cost_weight=float(
                self._p("tracking.point_count_cost_weight")
            ),
            mask_iou_cost_weight=float(
                self._p("tracking.mask_iou_cost_weight")
            ),
        )
        self._tracker = PointCloudTracker(self._tracker_config)
        self._intrusion_gate = SuddenIntrusionGate(
            SuddenIntrusionGateConfig(
                enabled=bool(self._p("hand_candidate.enabled")),
                minimum_seed_speed=float(
                    self._p("hand_candidate.minimum_seed_speed_mps")
                ),
                minimum_motion_hits=int(
                    self._p("hand_candidate.minimum_motion_hits")
                ),
                maximum_seed_age_frames=int(
                    self._p("hand_candidate.maximum_seed_age_frames")
                ),
            )
        )
        self._hand_semantic_enabled = bool(
            self._p("hand_semantics.enabled")
        )
        self._hand_semantic_gate = HandSemanticGate(
            HandSemanticGateConfig(
                minimum_confidence=float(
                    self._p(
                        "hand_semantics.minimum_detection_confidence"
                    )
                ),
                minimum_box_iou=float(
                    self._p("hand_semantics.minimum_box_iou")
                ),
                minimum_projection_coverage=float(
                    self._p(
                        "hand_semantics.minimum_projection_coverage"
                    )
                ),
                minimum_positive_point_coverage=float(
                    self._p(
                        "hand_semantics.minimum_positive_point_coverage"
                    )
                ),
                require_segmentation_mask=True,
                fail_closed_on_detector_unavailable=True,
            )
        )
        self._hand_semantic_hold_frames = max(
            int(self._p("hand_semantics.hold_frames")), 0
        )
        self._hand_semantic_holds: dict[int, int] = {}
        # True while the hand-only output is intentionally not being updated.
        # The transition guard prevents repeatedly resetting the asynchronous
        # EdgeTAM stream on every bad frame.
        self._semantic_output_hold_active = False
        self._hand_semantic_status = (
            "loading" if self._hand_semantic_enabled else "disabled"
        )
        self._hand_semantic_error = ""
        self._hand_detector: MediaPipeHandDetector | None = None
        if self._hand_semantic_enabled:
            try:
                self._hand_detector = MediaPipeHandDetector(
                    MediaPipeHandDetectorConfig(
                        maximum_hands=int(
                            self._p("hand_semantics.maximum_hands")
                        ),
                        minimum_detection_confidence=float(
                            self._p(
                                "hand_semantics.minimum_detection_confidence"
                            )
                        ),
                        minimum_tracking_confidence=float(
                            self._p(
                                "hand_semantics.minimum_tracking_confidence"
                            )
                        ),
                        model_complexity=int(
                            self._p("hand_semantics.model_complexity")
                        ),
                        mask_padding_pixels=int(
                            self._p("hand_semantics.mask_padding_pixels")
                        ),
                        static_image_mode=bool(
                            self._p("hand_semantics.static_image_mode")
                        ),
                        rotation_augmentation_degrees=tuple(
                            int(value)
                            for value in self._p(
                                "hand_semantics.rotation_augmentation_degrees"
                            )
                        ),
                        cycle_rotation_augmentation=bool(
                            self._p(
                                "hand_semantics.cycle_rotation_augmentation"
                            )
                        ),
                        temporal_hold_frames=int(
                            self._p("hand_semantics.temporal_hold_frames")
                        ),
                        temporal_confidence_decay=float(
                            self._p(
                                "hand_semantics.temporal_confidence_decay"
                            )
                        ),
                        minimum_flow_points=int(
                            self._p("hand_semantics.minimum_flow_points")
                        ),
                        maximum_flow_error=float(
                            self._p("hand_semantics.maximum_flow_error")
                        ),
                    )
                )
                self._hand_detector.load()
                self._hand_semantic_status = "ready"
            except Exception as exc:
                self._hand_detector = None
                self._hand_semantic_status = "unavailable"
                self._hand_semantic_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.get_logger().error(
                    "Hand semantic detector unavailable; strict hand-only "
                    "output will remain held: "
                    f"{self._hand_semantic_error}"
                )
        self._tracking_enabled = bool(self._p("tracking.enabled"))
        if not bool(self._p("edgetam.prompt_confirmed_tracks_only")):
            raise ValueError(
                "edgetam.prompt_confirmed_tracks_only must remain true: "
                "TENTATIVE noise may not create an EdgeTAM object"
            )

        self._projection_config = ProjectionConfig(
            box_padding_pixels=int(self._p("projection.box_padding_px")),
            projection_dilation_pixels=int(
                self._p("projection.dilation_pixels")
            ),
            maximum_positive_points=int(
                self._p("projection.positive_point_count")
            ),
            generate_negative_points=int(
                self._p("projection.negative_point_count")
            )
            > 0,
            maximum_negative_points=int(
                self._p("projection.negative_point_count")
            ),
        )
        morphology = max(
            int(self._p("fusion.mask_erode_pixels")),
            int(self._p("fusion.mask_dilate_pixels")),
            1,
        )
        self._fusion_config = FusionConfig(
            depth_axis=int(self._p("clustering.depth_axis")),
            morphology_kernel_size=morphology * 2 + 1,
            erosion_iterations=int(self._p("fusion.mask_erode_pixels")) > 0,
            dilation_iterations=int(self._p("fusion.mask_dilate_pixels")) > 0,
            aabb_gate_margin=float(
                self._p("fusion.spatial_gate_padding_m")
            ),
            absolute_depth_gate=float(self._p("fusion.depth_gate_m")),
            minimum_fused_points=int(self._p("fusion.minimum_fused_points")),
        )
        self._pointcloud_quality_config = PointCloudQualityConfig(
            depth_axis=int(self._p("clustering.depth_axis")),
            minimum_valid_points=max(
                min(int(self._p("clustering.min_points")), 6), 3
            ),
            minimum_good_points=max(
                int(self._p("clustering.min_points")) * 2, 20
            ),
        )
        self._mask_quality_config = MaskQualityConfig(
            minimum_mask_pixels=max(
                int(self._p("fusion.minimum_fused_points")), 3
            ),
            good_score_threshold=float(
                self._p("quality.mask_good_threshold")
            ),
            degraded_score_threshold=float(
                self._p("quality.mask_degraded_threshold")
            ),
            minimum_good_valid_depth_ratio=float(
                self._p("fusion.minimum_valid_depth_ratio")
            ),
            minimum_valid_depth_ratio=float(
                self._p("fusion.minimum_valid_depth_ratio")
            ),
            minimum_valid_depth_points=max(
                int(self._p("fusion.minimum_fused_points")), 3
            ),
            model_score_weight=float(self._p("quality.mask_score_weight")),
            projection_coverage_weight=float(
                self._p("quality.projected_cluster_coverage_weight")
            ),
            mask_cluster_iou_weight=float(
                self._p("quality.mask_cluster_iou_weight")
            ),
            valid_depth_weight=float(
                self._p("quality.valid_depth_ratio_weight")
            ),
            temporal_iou_weight=float(
                self._p("quality.temporal_mask_iou_weight")
            ),
            prediction_consistency_weight=float(
                self._p("quality.prediction_consistency_weight")
            ),
            area_consistency_weight=float(
                self._p("quality.area_change_weight")
            ),
        )
        max_area_ratio = float(
            self._p("fusion.maximum_mask_area_change_ratio")
        )
        self._reprompt_config = RepromptConfig(
            minimum_mask_cluster_iou=float(
                self._p("fusion.minimum_mask_cluster_iou")
            ),
            minimum_valid_depth_ratio=float(
                self._p("fusion.minimum_valid_depth_ratio")
            ),
            minimum_area_ratio=1.0 / max(max_area_ratio, 1.0),
            maximum_area_ratio=max(max_area_ratio, 1.0),
            maximum_centroid_error=float(
                self._p("fusion.maximum_centroid_difference_m")
            ),
            maximum_frames_without_cluster_points=int(
                self._p("fusion.re_prompt_after_invalid_frames")
            ),
        )
        self._confidence_config = ConfidenceConfig(
            pointcloud_weight=float(self._p("quality.pointcloud_weight")),
            temporal_tracking_weight=float(
                self._p("quality.temporal_tracking_weight")
            ),
            mask_consistency_weight=float(
                self._p("quality.mask_consistency_weight")
            ),
            near_obstacle_distance=float(
                self._p("safety.emergency_distance_m")
            ),
            base_uncertainty_margin=float(
                self._p("safety.base_uncertainty_margin_m")
            ),
        )

        safety_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        diagnostics_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._obstacles_publisher = self.create_publisher(
            TrackedObstacleArray,
            str(self._p("topics.output_obstacles")),
            safety_qos,
        )
        self._cloud_publisher = self.create_publisher(
            PointCloud2,
            str(self._p("topics.output_obstacle_cloud")),
            safety_qos,
        )
        self._legacy_cloud_publisher = None
        if bool(self._p("compatibility.publish_legacy_obstacle_alias")):
            self._legacy_cloud_publisher = self.create_publisher(
                PointCloud2,
                str(self._p("topics.output_legacy_obstacle_cloud")),
                safety_qos,
            )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            str(self._p("topics.output_diagnostics")),
            diagnostics_qos,
        )
        self._fps_publisher = self.create_publisher(
            Float32, str(self._p("topics.output_fps")), diagnostics_qos
        )
        self._latency_publisher = self.create_publisher(
            Float32,
            str(self._p("topics.output_latency_ms")),
            diagnostics_qos,
        )
        self._debug_image_publisher = None
        self._debug_cloud_publisher = None
        self._self_filtered_publisher = None
        self._markers_publisher = None
        if bool(self._p("performance.publish_debug_image")):
            self._debug_image_publisher = self.create_publisher(
                Image,
                str(self._p("topics.output_debug_image")),
                qos_profile_sensor_data,
            )
        if bool(self._p("performance.publish_debug_cloud")):
            self._debug_cloud_publisher = self.create_publisher(
                PointCloud2,
                str(self._p("topics.output_debug_cloud")),
                qos_profile_sensor_data,
            )
            self._self_filtered_publisher = self.create_publisher(
                PointCloud2,
                str(self._p("topics.output_self_filtered_cloud")),
                qos_profile_sensor_data,
            )
        if bool(self._p("performance.publish_markers")):
            self._markers_publisher = self.create_publisher(
                MarkerArray,
                str(self._p("topics.output_markers")),
                diagnostics_qos,
            )

        queue_length = max(
            int(self._p("performance.maximum_queue_length")), 1
        )
        if bool(self._p("performance.use_latest_frame_only")):
            queue_length = 1
        self._work_queue: queue.Queue[SensorBundle | None] = queue.Queue(
            maxsize=queue_length
        )
        self._shutdown = threading.Event()
        self._state_lock = threading.Lock()
        self._publication_lock = threading.RLock()
        self._edge_context_lock = threading.RLock()
        self._dropped_bundles = 0
        self._last_geometry_received = 0.0
        self._last_success = 0.0
        self._last_processed_stamp: float | None = None
        self._frame_index = 0
        self._latest_fps = 0.0
        self._latest_latency_ms = 0.0
        self._latest_edge_latency_ms = 0.0
        self._recent_completion_times: list[float] = []
        self._last_prediction_publish = 0.0
        self._geometry_gap_active = False
        self._pipeline_level = DiagnosticStatus.WARN
        self._pipeline_message = "waiting for geometry"
        self._diagnostic_values: dict[str, str] = {}
        self._latest_prompts: dict[int, ProjectionPrompt] = {}
        self._latest_masks: dict[int, MaskObservation] = {}
        self._latest_mask_diagnostics: dict[str, str] = {
            "mask_count": "0",
            "mask_good_count": "0",
            "mask_degraded_count": "0",
            "mask_invalid_count": "0",
            "mask_reject_reasons": "",
        }
        self._oversize_prompt_rejections = 0
        self._previous_masks: dict[int, np.ndarray] = {}
        self._previous_mask_prompts: dict[int, ProjectionPrompt] = {}
        self._reprompt_reasons: dict[int, str] = {}
        self._last_reprompt_stamp: dict[int, float] = {}
        self._frames_without_cluster_points: dict[int, int] = {}
        self._last_edge_result_sequence = 0
        self._edge_context_generation = 0
        self._edge_context_limit = max(
            int(self._p("edgetam.rolling_window_frames")) * 2,
            8,
        )
        self._edge_contexts: OrderedDict[int, _EdgeFrameContext] = (
            OrderedDict()
        )
        self._safety_publication_serial = 0
        self._last_safety_output_stamp: float | None = None
        self._edge_refined_corrections = 0
        self._edge_stale_results = 0
        self._edge_enabled = bool(self._p("edgetam.enabled"))
        self._edge_status = (
            "loading" if self._edge_enabled else "disabled"
        )
        self._edge_error = ""
        self._edge: EdgeTAMWrapper | None = None
        self._edge_loader: threading.Thread | None = None
        set_enabled_service = str(
            self._p("topics.set_enabled_service")
        ).strip()
        if not set_enabled_service:
            raise ValueError("topics.set_enabled_service must not be empty")
        self._set_enabled_service = self.create_service(
            SetBool,
            set_enabled_service,
            self._set_edgetam_enabled,
        )

        self._synchronizer = RosSensorSynchronizer(
            self,
            rgb_topic=str(self._p("topics.rgb_image")),
            depth_topic=str(self._p("topics.depth_image")),
            camera_info_topic=str(self._p("topics.camera_info")),
            pointcloud_topic=str(self._p("topics.pointcloud")),
            queue_size=int(self._p("sync.queue_size")),
            slop_sec=float(self._p("sync.slop_sec")),
            fallback_delay_sec=float(
                self._p("sync.pointcloud_fallback_delay_sec")
            ),
            callback=self._enqueue_bundle,
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="pointcloud-tracker-latest-worker",
            daemon=True,
        )
        self._worker.start()
        if self._edge_enabled:
            self._ensure_edgetam_loader()

        diagnostics_rate = max(
            float(self._p("performance.diagnostics_rate_hz")), 0.2
        )
        self._diagnostics_timer = self.create_timer(
            1.0 / diagnostics_rate, self._publish_diagnostics
        )
        self._stale_timer = self.create_timer(0.1, self._check_stale)
        self.get_logger().info(
            "Point-cloud-first hand tracker started; only RGB+3D confirmed "
            "hands may reach the obstacle output."
        )

    def _p(self, name: str) -> Any:
        return self.get_parameter(name).value

    def _plane_normal_axis_parameter(self) -> np.ndarray | None:
        """Return an optional normalized-axis input for background planes.

        An empty parameter keeps generic RANSAC behavior.  A three-element
        value is used by fixed-camera profiles to remove only a dominant plane
        facing the camera, rather than accidentally treating a hand or another
        small planar obstacle as background.
        """

        values = list(self._p("pointcloud.plane_normal_axis"))
        if not values:
            return None
        if len(values) != 3:
            raise ValueError(
                "pointcloud.plane_normal_axis must be empty or contain x/y/z"
            )
        axis = np.asarray(values, dtype=np.float32)
        if not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 1e-12:
            raise ValueError(
                "pointcloud.plane_normal_axis must be a finite non-zero vector"
            )
        return axis

    def _set_edgetam_enabled(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """Enable/disable only RGB mask refinement, never 3D tracking.

        Disabling invalidates every queued exact-frame context and resets the
        wrapper's rolling video state.  The loaded predictor is deliberately
        retained in memory so a later enable does not create a blind model
        loading interval or disturb the point-cloud safety path.
        """

        requested = bool(request.data)
        if self._shutdown.is_set():
            response.success = False
            response.message = "node is shutting down"
            return response

        if not requested:
            with self._edge_context_lock:
                was_enabled = self._edge_enabled
                self._edge_enabled = False
                self._edge_status = "disabled"
                self._edge_error = ""
            # Invalidate outside the state update so an in-flight result can
            # only fail its generation/publication gate; reset_stream is
            # serialized by the wrapper itself.
            self._invalidate_edge_contexts(reset_wrapper=True)
            response.success = True
            response.message = (
                "EdgeTAM refinement disabled; point-cloud tracking remains active"
                if was_enabled
                else "EdgeTAM refinement is already disabled"
            )
            self.get_logger().info(response.message)
            return response

        with self._edge_context_lock:
            was_enabled = self._edge_enabled
            self._edge_enabled = True
            edge_ready = self._edge is not None and self._edge.available
            if edge_ready:
                self._edge_status = "ready"
                self._edge_error = ""
            else:
                self._edge_status = "loading"
                self._edge_error = ""
        if not was_enabled:
            # Never reuse masks, prompts, or an official video state that was
            # produced before the runtime gate was re-enabled.
            self._invalidate_edge_contexts(reset_wrapper=True)
        if not edge_ready:
            self._ensure_edgetam_loader()
        response.success = True
        response.message = (
            "EdgeTAM refinement enabled"
            if edge_ready
            else "EdgeTAM enable accepted; model is loading asynchronously"
        )
        self.get_logger().info(response.message)
        return response

    def _ensure_edgetam_loader(self) -> None:
        """Start at most one asynchronous model loader when enabled."""

        with self._edge_context_lock:
            if self._shutdown.is_set() or not self._edge_enabled:
                return
            if self._edge is not None and self._edge.available:
                self._edge_status = "ready"
                self._edge_error = ""
                return
            if self._edge_loader is not None and self._edge_loader.is_alive():
                self._edge_status = "loading"
                return
            self._edge_status = "loading"
            self._edge_error = ""
            loader = threading.Thread(
                target=self._load_edgetam,
                name="edgetam-loader",
                daemon=True,
            )
            self._edge_loader = loader
            # Start while holding the lock so two executor callbacks cannot
            # both observe a not-yet-alive thread and launch duplicate models.
            loader.start()

    def _set_edge_inference_status(
        self, status: str, error: str = ""
    ) -> bool:
        """Update inference status only while the runtime gate is enabled."""

        with self._edge_context_lock:
            if not self._edge_enabled:
                return False
            self._edge_status = str(status)
            self._edge_error = str(error)
            return True

    def _enqueue_bundle(self, bundle: SensorBundle) -> None:
        if bundle.pointcloud is None and bundle.depth is None:
            return
        while True:
            try:
                self._work_queue.put_nowait(bundle)
                return
            except queue.Full:
                try:
                    self._work_queue.get_nowait()
                    self._work_queue.task_done()
                    with self._state_lock:
                        self._dropped_bundles += 1
                except queue.Empty:
                    return

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                bundle = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                self._poll_edge_result()
                self._maybe_publish_prediction()
                continue
            if bundle is None:
                self._work_queue.task_done()
                break
            started = time.perf_counter()
            try:
                self._process_bundle(bundle, started)
            except Exception as exc:  # fail closed at the ROS adapter boundary
                self.get_logger().error(
                    f"Geometry frame rejected: {type(exc).__name__}: {exc}"
                )
                self._set_pipeline_status(
                    DiagnosticStatus.ERROR,
                    "geometry frame rejected; prior output is not replaced",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._work_queue.task_done()

    def _process_bundle(
        self, bundle: SensorBundle, started: float
    ) -> None:
        # Consume a completed result before a newer geometry publication. This
        # permits an exact same-stamp correction without ever delaying the
        # point-cloud-first path or allowing an older result to overwrite it.
        self._poll_edge_result()
        geometry_message = (
            bundle.pointcloud
            if bundle.pointcloud is not None
            else bundle.depth
        )
        if geometry_message is None:
            return
        rgb: np.ndarray | None = None
        rgb_stamp: float | None = None
        rgb_decode_error = ""
        if bundle.rgb is not None:
            try:
                rgb = self._normalize_rgb(
                    image_message_to_array(bundle.rgb)
                )
                rgb_stamp = stamp_to_seconds(bundle.rgb)
            except Exception as exc:
                rgb_decode_error = (
                    f"{type(exc).__name__}: {exc}"
                )
        geometry_fallback = ""
        if bundle.pointcloud is not None:
            # Never infer RGB alignment from equal array dimensions. Exact
            # camera projection is attached later using CameraInfo + TF.
            try:
                sensor_cloud = pointcloud2_to_cloud(bundle.pointcloud)
                stamp = stamp_to_seconds(bundle.pointcloud)
                if len(sensor_cloud.points) == 0:
                    raise ValueError(
                        "PointCloud2 contains no finite points"
                    )
            except Exception as pointcloud_exc:
                try:
                    if bundle.depth is None:
                        raise ValueError("depth fallback is absent")
                    geometry_message = bundle.depth
                    stamp = stamp_to_seconds(bundle.depth)
                    sensor_cloud = self._depth_cloud_from_bundle(
                        bundle,
                        stamp=stamp,
                    )
                    geometry_fallback = (
                        "depth_after_pointcloud_rejection: "
                        f"{type(pointcloud_exc).__name__}: "
                        f"{pointcloud_exc}"
                    )
                except Exception as depth_exc:
                    self._set_pipeline_status(
                        DiagnosticStatus.ERROR,
                        "PointCloud2 rejected and no valid depth fallback; "
                        "prior output held",
                        pointcloud_error=(
                            f"{type(pointcloud_exc).__name__}: "
                            f"{pointcloud_exc}"
                        ),
                        depth_fallback_error=(
                            f"{type(depth_exc).__name__}: {depth_exc}"
                        ),
                    )
                    return
        else:
            try:
                assert bundle.depth is not None
                stamp = stamp_to_seconds(bundle.depth)
                sensor_cloud = self._depth_cloud_from_bundle(
                    bundle,
                    stamp=stamp,
                )
            except Exception as exc:
                self._set_pipeline_status(
                    DiagnosticStatus.ERROR,
                    "depth CameraInfo context invalid; prior output held",
                    camera_info_error=f"{type(exc).__name__}: {exc}",
                )
                return

        # Validate the timestamp of the geometry actually selected above.
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        validation = validate_timestamps(
            [stamp],
            slop_sec=float(self._p("sync.slop_sec")),
            now_sec=now_sec if now_sec > 0.0 else None,
            max_data_age_sec=float(self._p("sync.max_data_age_sec")),
        )
        if not validation.valid:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                f"sensor timestamp rejected: {validation.reason}",
                synchronization_span_sec=f"{validation.span_sec:.6f}",
            )
            return
        clock_reset = False
        if (
            self._last_processed_stamp is not None
            and stamp <= self._last_processed_stamp
        ):
            regression = self._last_processed_stamp - stamp
            if regression <= 1e-9:
                self._set_pipeline_status(
                    DiagnosticStatus.WARN,
                    "duplicate geometry timestamp dropped",
                    duplicate_stamp=f"{stamp:.9f}",
                )
                return
            reset_threshold = max(
                2.0 * float(self._p("sync.sensor_stale_timeout_sec")),
                1.0,
            )
            if regression <= reset_threshold:
                # Delayed fallback callbacks are stale measurements, not a
                # bag/clock reset, and must not churn every track ID.
                self._set_pipeline_status(
                    DiagnosticStatus.WARN,
                    "out-of-order geometry bundle dropped",
                    timestamp_regression_sec=f"{regression:.6f}",
                )
                return
            clock_reset = True

        if rgb is not None and rgb_stamp is not None:
            rgb_validation = validate_timestamps(
                [rgb_stamp],
                slop_sec=float(self._p("sync.slop_sec")),
                now_sec=now_sec if now_sec > 0.0 else None,
                max_data_age_sec=float(self._p("sync.max_data_age_sec")),
            )
            rgb_delta = abs(rgb_stamp - stamp)
            sync_slop = max(float(self._p("sync.slop_sec")), 0.0)
            if not rgb_validation.valid or rgb_delta > sync_slop:
                rgb_decode_error = (
                    "RGB context rejected: "
                    + (
                        rgb_validation.reason
                        if not rgb_validation.valid
                        else (
                            f"RGB/geometry delta {rgb_delta:.6f}s > "
                            f"{sync_slop:.6f}s"
                        )
                    )
                )
                rgb = None
                rgb_stamp = None
        if len(sensor_cloud.points) == 0:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "geometry frame contains no finite depth/points; prior output held",
            )
            return
        if clock_reset:
            self.get_logger().warning(
                "Sensor clock reset detected; resetting temporal state"
            )
            self._tracker.reset()
            self._intrusion_gate.reset()
            self._hand_semantic_holds.clear()
            self._invalidate_edge_contexts(reset_wrapper=True)
            self._reset_safety_publication_epoch()
            self._frame_index = 0
        self._last_processed_stamp = stamp
        with self._state_lock:
            self._last_geometry_received = time.monotonic()
        if not sensor_cloud.frame_id:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "geometry frame_id is empty; output held",
            )
            return
        try:
            tracking_cloud = self._cloud_in_tracking_frame(
                sensor_cloud, geometry_message.header.stamp
            )
            robot_origin = self._frame_origin(
                self._base_frame, geometry_message.header.stamp
            )
        except TransformException as exc:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "required TF unavailable; output held",
                tf_error=str(exc),
            )
            return
        self._tracker.set_robot_origin(robot_origin)
        self._fusion_config.robot_origin = tuple(
            float(value) for value in robot_origin
        )

        link_spheres, link_error = self._link_spheres(
            geometry_message.header.stamp
        )
        self_filter_result = self._self_filter.filter(
            tracking_cloud, link_spheres
        )
        if (
            self_filter_result.status is SelfFilterStatus.UNAVAILABLE
            and not self_filter_result.safe_to_continue
        ):
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "robot self-filter unavailable; fail-closed output hold",
                self_filter_reason=(
                    link_error or self_filter_result.reason
                ),
            )
            return
        with self._state_lock:
            resumed_after_gap = self._geometry_gap_active
            if resumed_after_gap:
                self._geometry_gap_active = False
        if resumed_after_gap:
            self._invalidate_edge_contexts(reset_wrapper=True)

        preprocessing_started = time.perf_counter()
        # Preserve a pre-background, workspace-cropped cloud for mask-guided
        # recovery of weak/contact hand pixels. It is never used to create a
        # cluster or prompt by itself; the subtracted cloud remains the strong
        # geometric seed.
        fusion_raw_cloud = self._preprocessor.workspace_cloud(
            self_filter_result.filtered_cloud
        )
        background_result = self._background_filter.filter(
            self_filter_result.filtered_cloud
        )
        background_trusted = _background_output_is_trusted(
            background_result.state,
            background_enabled=self._background_filter.config.enabled,
            ray_depth_enabled=(
                self._background_filter.config.ray_depth_enabled
            ),
            alignment_valid=background_result.alignment_valid,
        )
        recovery_hand_detections: list[HandDetection] | None = None
        recovery_tracking_to_camera: np.ndarray | None = None
        recovery_projection_error = ""
        if (
            not background_trusted
            and background_result.state is not StaticBackgroundState.READY
        ):
            # Hand-only means exactly that: an unfinished or untrusted
            # background model cannot turn the entire fixed scene into a hand
            # obstacle.  Do not publish an empty cloud either, because that
            # would falsely assert that perception proved the scene clear.
            # Holding publication lets the relationship/output watchdog make
            # the controller STOP on stale perception.
            calibration_active = background_result.state in {
                StaticBackgroundState.WARMING_UP,
                StaticBackgroundState.CALIBRATING,
            }
            if calibration_active:
                self._tracker.reset()
                self._intrusion_gate.reset()
                self._hand_semantic_holds.clear()
            if not self._semantic_output_hold_active:
                self._invalidate_edge_contexts(reset_wrapper=True)
            self._semantic_output_hold_active = True
            hold_reason = (
                "background calibration active; hand-only safety output held"
                if calibration_active
                else (
                    "background alignment untrusted; hand-only safety output "
                    "held"
                )
            )
            self._set_pipeline_status(
                DiagnosticStatus.WARN,
                hold_reason,
                input_mode=str(self._p("input_mode")),
                synchronized=str(bundle.synchronized),
                sync_reason=bundle.reason,
                stamp=f"{stamp:.9f}",
                input_points=str(len(sensor_cloud.points)),
                self_filtered_points=str(
                    len(self_filter_result.filtered_cloud.points)
                ),
                processed_points="0",
                cluster_count="0",
                track_count="0",
                hand_candidate_count="0",
                geometry_fallback_track_count="0",
                hand_semantic_enabled=str(
                    self._hand_semantic_enabled
                ).lower(),
                hand_semantic_status=self._hand_semantic_status,
                hand_semantic_error=self._hand_semantic_error,
                hand_rgb_detection_count="0",
                hand_semantic_reject_reasons="",
                prompt_count="0",
                safety_output_state="held_untrusted_background",
                background_state=background_result.state.value,
                background_removed=str(background_result.removed_points),
                background_baseline_points=str(
                    background_result.baseline_points
                ),
                background_matched_points=str(
                    background_result.matched_points
                ),
                background_alignment_points=str(
                    background_result.alignment_points
                ),
                background_depth_scale=(
                    f"{background_result.depth_scale:.5f}"
                ),
                background_candidate_depth_scale=(
                    f"{background_result.alignment_candidate_scale:.5f}"
                ),
                background_candidate_alignment_points=str(
                    background_result.alignment_candidate_points
                ),
                background_candidate_alignment_reason=(
                    background_result.alignment_candidate_reason
                ),
                background_alignment_valid=str(
                    background_result.alignment_valid
                ).lower(),
                background_warmup=(
                    f"{background_result.warmup_progress}/"
                    f"{self._background_filter.config.warmup_frames}"
                ),
                background_calibration=(
                    f"{background_result.calibration_progress}/"
                    f"{self._background_filter.config.calibration_frames}"
                ),
            )
            self._publish_live_debug_snapshot(
                rgb,
                None if bundle.rgb is None else bundle.rgb.header,
                status_text=(
                    f"LIVE frame={self._frame_index} · {hold_reason} · no mask"
                ),
            )
            self._record_processing_metrics(started, output_published=False)
            return
        processing_cloud = background_result.cloud
        if (
            background_result.state is StaticBackgroundState.READY
            and self._hand_semantic_enabled
            and self._hand_detector is not None
            and rgb is not None
            and bundle.rgb is not None
            and bundle.camera_info is not None
        ):
            # Seed geometry only from the actual synchronized RGB hand mask
            # and aligned depth on every frame.  Background-motion points are
            # deliberately excluded: joining them to a valid hand mask can
            # turn a moving table/robot return into one oversized obstacle.
            # Detector/projection failures therefore remain fail-closed.
            processing_cloud = fusion_raw_cloud.select(
                np.zeros(len(fusion_raw_cloud.points), dtype=bool)
            )
            try:
                assert rgb_stamp is not None
                rgb_geometry_delta = abs(rgb_stamp - stamp)
                sync_slop = max(float(self._p("sync.slop_sec")), 0.0)
                if rgb_geometry_delta > sync_slop:
                    raise ValueError(
                        "RGB/geometry timestamp mismatch: "
                        f"{rgb_geometry_delta:.6f}s > {sync_slop:.6f}s"
                    )
                camera_info_error = self._camera_info_validation_error(
                    bundle.camera_info,
                    bundle.rgb,
                    rgb.shape[:2],
                )
                if camera_info_error:
                    raise ValueError(camera_info_error)
                recovery_tracking_to_camera = (
                    self._tracking_to_camera_matrix(
                        bundle.camera_info,
                        bundle.rgb,
                        bundle.rgb.header.stamp,
                    )
                )
                fusion_raw_cloud = self._cloud_with_camera_pixels(
                    fusion_raw_cloud,
                    bundle.camera_info,
                    rgb.shape[:2],
                    recovery_tracking_to_camera,
                )
                recovery_hand_detections = (
                    self._hand_detector.infer_rgb(rgb)
                )
                rgb_hand_cloud = self._cloud_inside_hand_detections(
                    fusion_raw_cloud,
                    recovery_hand_detections,
                    rgb.shape[:2],
                    tracking_to_camera=recovery_tracking_to_camera,
                    foreground_depth_quantile=float(
                        self._p("hand_semantics.foreground_depth_quantile")
                    ),
                    foreground_depth_span_m=float(
                        self._p("hand_semantics.foreground_depth_span_m")
                    ),
                )
                processing_cloud = rgb_hand_cloud
            except (ValueError, TransformException) as exc:
                recovery_projection_error = f"{type(exc).__name__}: {exc}"
        preprocessed = self._preprocessor.process(
            processing_cloud
        )
        preprocessing_ms = (
            time.perf_counter() - preprocessing_started
        ) * 1000.0
        clustering_started = time.perf_counter()
        clusters = self._clusterer.extract(
            preprocessed.processed_cloud, robot_origin=robot_origin
        )
        clusters.sort(key=lambda item: item.nearest_distance)
        clusters = clusters[
            : max(int(self._p("clustering.maximum_clusters")), 1)
        ]
        if bool(
            self._p("pointcloud.use_high_resolution_for_final_geometry")
        ):
            clusters = [
                self._refine_cluster(
                    cluster, preprocessed.raw_cloud, robot_origin
                )
                for cluster in clusters
            ]
        previous_tracks = self._tracker.predict_to(stamp)
        previous_states = {
            track.track_id: track.state for track in previous_tracks
        }
        for cluster in clusters:
            if previous_tracks:
                distances = [
                    float(
                        np.linalg.norm(
                            cluster.centroid - previous.position
                        )
                    )
                    for previous in previous_tracks
                ]
                nearest_previous = previous_tracks[int(np.argmin(distances))]
                cluster.innovation_distance = min(distances)
                previous_aabb = nearest_previous.aabb
            else:
                previous_aabb = None
            apply_pointcloud_quality(
                cluster,
                previous_aabb=previous_aabb,
                config=self._pointcloud_quality_config,
            )
        usable_clusters, emergency_invalid_clusters = (
            self._select_usable_clusters(
                clusters,
                float(self._p("safety.emergency_distance_m")),
            )
        )
        clustering_ms = (
            time.perf_counter() - clustering_started
        ) * 1000.0

        tracking_started = time.perf_counter()
        if not self._tracking_enabled:
            self._tracker.reset()
            self._intrusion_gate.reset()
            self._hand_semantic_holds.clear()
        all_tracks = self._tracker.update(usable_clusters, stamp)
        all_tracks = [
            track
            for track in all_tracks
            if track.state is not TrackingState.DELETED
        ]
        semantic_filter_active = True
        if recovery_hand_detections is not None:
            # These clusters already consist exclusively of calibrated 3D
            # rays inside an RGB hand mask. Requiring a second motion seed
            # would make a stationary hand disappear during background-scale
            # recovery. The tracker confirmation and the RGB/3D overlap gate
            # below remain mandatory before publication or EdgeTAM prompting.
            tracks = [
                track
                for track in all_tracks
                if track.state
                in {TrackingState.CONFIRMED, TrackingState.OCCLUDED}
            ]
        else:
            tracks = self._intrusion_gate.filter_tracks(
                all_tracks,
                baseline_ready=True,
            )
        for track in tracks:
            previous_state = previous_states.get(track.track_id)
            if (
                previous_state
                in {TrackingState.OCCLUDED, TrackingState.LOST}
                and track.state is TrackingState.CONFIRMED
            ):
                self._reprompt_reasons[track.track_id] = (
                    "reappeared_after_occlusion"
                )
                self._last_reprompt_stamp.pop(track.track_id, None)
            if resumed_after_gap and track.state is TrackingState.CONFIRMED:
                self._reprompt_reasons[track.track_id] = (
                    "geometry_gap_recovery"
                )
                self._last_reprompt_stamp.pop(track.track_id, None)
        tracking_ms = (time.perf_counter() - tracking_started) * 1000.0

        self._frame_index += 1
        prompts: dict[int, ProjectionPrompt] = {}
        projection_error = rgb_decode_error or recovery_projection_error
        tracking_to_camera: np.ndarray | None = recovery_tracking_to_camera
        hand_detection_count = (
            0
            if recovery_hand_detections is None
            else len(recovery_hand_detections)
        )
        hand_semantic_reject_reasons = ""
        geometry_safety_fallback = False
        hand_output_hold_reason = ""
        if rgb is not None and bundle.camera_info is not None:
            try:
                assert rgb_stamp is not None
                rgb_geometry_delta = abs(rgb_stamp - stamp)
                sync_slop = max(float(self._p("sync.slop_sec")), 0.0)
                if rgb_geometry_delta > sync_slop:
                    raise ValueError(
                        "RGB/geometry timestamp mismatch: "
                        f"{rgb_geometry_delta:.6f}s > {sync_slop:.6f}s"
                    )
                camera_info_error = self._camera_info_validation_error(
                    bundle.camera_info,
                    bundle.rgb,
                    rgb.shape[:2],
                )
                if camera_info_error:
                    raise ValueError(camera_info_error)
                intrinsics = camera_intrinsics_from_info(
                    bundle.camera_info
                )
                if (intrinsics.height, intrinsics.width) != rgb.shape[:2]:
                    raise ValueError(
                        "CameraInfo/RGB shape mismatch: "
                        f"{intrinsics.width}x{intrinsics.height} vs "
                        f"{rgb.shape[1]}x{rgb.shape[0]}"
                    )
                tracking_to_camera = self._tracking_to_camera_matrix(
                    bundle.camera_info,
                    bundle.rgb,
                    bundle.rgb.header.stamp,
                )
                preprocessed.raw_cloud = self._cloud_with_camera_pixels(
                    preprocessed.raw_cloud,
                    bundle.camera_info,
                    rgb.shape[:2],
                    tracking_to_camera,
                )
                fusion_raw_cloud = self._cloud_with_camera_pixels(
                    fusion_raw_cloud,
                    bundle.camera_info,
                    rgb.shape[:2],
                    tracking_to_camera,
                )
                prompts = self._make_prompts(
                    tracks if semantic_filter_active else [],
                    bundle.camera_info,
                    rgb.shape[:2],
                    tracking_to_camera,
                )
                projection_error = ""
            except (ValueError, TransformException) as exc:
                projection_error = f"{type(exc).__name__}: {exc}"

        if semantic_filter_active:
            if self._hand_semantic_enabled:
                live_candidate_ids = {
                    track.track_id for track in tracks
                }
                held_only_context = bool(
                    live_candidate_ids
                    and live_candidate_ids
                    <= set(self._hand_semantic_holds)
                )
                semantic_context_valid = bool(
                    rgb is not None
                    and not projection_error
                    and self._hand_detector is not None
                    and (
                        not tracks
                        or held_only_context
                        or (
                            bool(prompts)
                            and tracking_to_camera is not None
                        )
                    )
                )
                if semantic_context_valid:
                    try:
                        assert rgb is not None
                        assert self._hand_detector is not None
                        detections = (
                            recovery_hand_detections
                            if recovery_hand_detections is not None
                            else self._hand_detector.infer_rgb(rgb)
                        )
                        hand_detection_count = len(detections)
                        accepted, decisions = (
                            self._hand_semantic_gate.filter_prompts(
                                prompts,
                                detections,
                                rgb.shape[:2],
                            )
                        )
                        self._hand_semantic_holds = {
                            track_id: remaining
                            for track_id, remaining
                            in self._hand_semantic_holds.items()
                            if track_id in live_candidate_ids
                        }
                        for track_id in live_candidate_ids:
                            if track_id in accepted:
                                self._hand_semantic_holds[track_id] = (
                                    self._hand_semantic_hold_frames
                                )
                            elif track_id in self._hand_semantic_holds:
                                remaining = (
                                    self._hand_semantic_holds[track_id] - 1
                                )
                                if remaining >= 0:
                                    self._hand_semantic_holds[track_id] = (
                                        remaining
                                    )
                                else:
                                    self._hand_semantic_holds.pop(
                                        track_id, None
                                    )
                        hand_ids = set(self._hand_semantic_holds)
                        tracks = [
                            replace(
                                track,
                                semantic_class="hand",
                                semantic_confirmed=True,
                            )
                            for track in tracks
                            if track.track_id in hand_ids
                        ]
                        # A short semantic hold may reuse the current exact 3D
                        # projection to let EdgeTAM bridge an RGB landmark miss.
                        prompts = {
                            track_id: prompt
                            for track_id, prompt in prompts.items()
                            if track_id in hand_ids
                        }
                        hand_semantic_reject_reasons = ",".join(
                            sorted(
                                {
                                    decision.reason
                                    for decision in decisions.values()
                                    if not decision.accepted
                                }
                            )
                        )
                        self._hand_semantic_status = "ready"
                        self._hand_semantic_error = ""
                        self._semantic_output_hold_active = False
                    except Exception as exc:
                        self._hand_semantic_status = "error"
                        self._hand_semantic_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._hand_semantic_holds.clear()
                        tracks = []
                        prompts = {}
                        hand_output_hold_reason = (
                            "RGB hand detector failed; hand-only safety "
                            "output held"
                        )
                else:
                    # A track that was already verified as a hand remains a
                    # conservative predicted obstacle through a bounded RGB /
                    # projection gap. Clearing it immediately made both the
                    # bounding box and controller obstacle flicker. New,
                    # unverified clusters still fail closed below.
                    held_ids = (
                        live_candidate_ids
                        & set(self._hand_semantic_holds)
                    )
                    if held_ids and held_ids == live_candidate_ids:
                        for track_id in tuple(held_ids):
                            remaining = (
                                self._hand_semantic_holds[track_id] - 1
                            )
                            if remaining >= 0:
                                self._hand_semantic_holds[track_id] = remaining
                            else:
                                self._hand_semantic_holds.pop(track_id, None)
                        held_ids = set(self._hand_semantic_holds)
                        tracks = [
                            replace(
                                track,
                                semantic_class="hand",
                                semantic_confirmed=True,
                            )
                            for track in tracks
                            if track.track_id in held_ids
                        ]
                        prompts = {
                            track_id: prompt
                            for track_id, prompt in prompts.items()
                            if track_id in held_ids
                        }
                        if held_ids:
                            self._hand_semantic_status = "tracking_held"
                            self._semantic_output_hold_active = False
                        else:
                            hand_output_hold_reason = (
                                "verified hand retention expired while RGB/3D "
                                "context unavailable; safety output held"
                            )
                    else:
                        # Never claim a new unverified cluster is a hand. The
                        # publication timeout remains the fail-closed signal.
                        self._hand_semantic_holds.clear()
                        tracks = []
                        prompts = {}
                        hand_output_hold_reason = (
                            "RGB/3D hand semantic context unavailable; "
                            "hand-only safety output held"
                        )
                        if self._hand_detector is None:
                            self._hand_semantic_status = "unavailable"
                        else:
                            self._hand_semantic_status = (
                                "waiting_for_rgb_projection"
                            )
            else:
                self._semantic_output_hold_active = False
                tracks = [
                    replace(
                        track,
                        semantic_class="hand_candidate",
                        semantic_confirmed=True,
                    )
                    for track in tracks
                ]

        if hand_output_hold_reason:
            if not self._semantic_output_hold_active:
                self._invalidate_edge_contexts(reset_wrapper=True)
            self._semantic_output_hold_active = True
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                hand_output_hold_reason,
                input_mode=str(self._p("input_mode")),
                synchronized=str(bundle.synchronized),
                sync_reason=bundle.reason,
                stamp=f"{stamp:.9f}",
                input_points=str(len(sensor_cloud.points)),
                self_filtered_points=str(
                    len(self_filter_result.filtered_cloud.points)
                ),
                processed_points=str(len(preprocessed.processed_cloud.points)),
                cluster_count=str(len(usable_clusters)),
                track_count="0",
                hand_candidate_count="0",
                geometry_fallback_track_count="0",
                hand_semantic_enabled="true",
                hand_semantic_status=self._hand_semantic_status,
                hand_semantic_error=self._hand_semantic_error,
                hand_rgb_detection_count=str(hand_detection_count),
                hand_semantic_reject_reasons=hand_semantic_reject_reasons,
                prompt_count="0",
                projection_error=projection_error,
                safety_output_state="held_untrusted_hand_semantics",
                background_state=background_result.state.value,
                background_removed=str(background_result.removed_points),
                background_baseline_points=str(
                    background_result.baseline_points
                ),
                background_matched_points=str(
                    background_result.matched_points
                ),
                background_alignment_points=str(
                    background_result.alignment_points
                ),
                background_depth_scale=(
                    f"{background_result.depth_scale:.5f}"
                ),
                background_candidate_depth_scale=(
                    f"{background_result.alignment_candidate_scale:.5f}"
                ),
                background_candidate_alignment_points=str(
                    background_result.alignment_candidate_points
                ),
                background_candidate_alignment_reason=(
                    background_result.alignment_candidate_reason
                ),
                background_alignment_valid=str(
                    background_result.alignment_valid
                ).lower(),
            )
            self._publish_live_debug_snapshot(
                rgb,
                None if bundle.rgb is None else bundle.rgb.header,
                status_text=(
                    f"LIVE frame={self._frame_index} · "
                    f"{hand_output_hold_reason} · no mask"
                ),
            )
            self._record_processing_metrics(started, output_published=False)
            return

        # The immediate publication is geometry-only. A completed EdgeTAM job
        # is fused later against the exact saved context for that sequence.
        masks: dict[int, MaskObservation] = {}
        fusion_started = time.perf_counter()
        output_tracks = self._decorate_and_fuse_tracks(
            tracks,
            usable_clusters,
            fusion_raw_cloud,
            prompts,
            masks,
            tracking_to_camera=tracking_to_camera,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

        self._latest_prompts = prompts
        if not prompts:
            with self._edge_context_lock:
                self._latest_mask_diagnostics = {
                    "mask_count": "0",
                    "mask_good_count": "0",
                    "mask_degraded_count": "0",
                    "mask_invalid_count": "0",
                    "mask_reject_reasons": "",
                }

        header = Header()
        header.stamp = geometry_message.header.stamp
        header.frame_id = self._tracking_frame
        publication_serial = self._publish_outputs(
            header,
            output_tracks,
            preprocessed.processed_cloud,
            self_filter_result.filtered_cloud,
            rgb,
            None if bundle.rgb is None else bundle.rgb.header,
            prompts,
            masks,
            recovery_hand_detections or [],
        )
        if publication_serial is None:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "non-monotonic safety publication rejected; prior output held",
                rejected_stamp=f"{stamp:.9f}",
            )
            return
        if (
            rgb is not None
            and prompts
        ):
            assert rgb_stamp is not None
            self._submit_edgetam(
                rgb,
                list(prompts.values()),
                rgb_stamp=rgb_stamp,
                geometry_stamp=stamp,
                header=header,
                rgb_header=bundle.rgb.header,
                tracks=tracks,
                clusters=usable_clusters,
                raw_cloud=fusion_raw_cloud,
                prompts_by_id=prompts,
                tracking_to_camera=tracking_to_camera,
                publication_serial=publication_serial,
            )
            # A fake/test predictor or a very fast device may already have
            # completed. This remains a non-blocking latest-result read.
            self._poll_edge_result()

        self._record_processing_metrics(started, output_published=True)
        with self._edge_context_lock:
            edge_status_value = self._edge_status
            edge_error_value = self._edge_error
        level = DiagnosticStatus.OK
        message = "point-cloud tracking active"
        if background_result.state in {
            StaticBackgroundState.WARMING_UP,
            StaticBackgroundState.CALIBRATING,
        }:
            level = DiagnosticStatus.WARN
            message = (
                "background calibration active; keep workspace empty"
            )
        elif (
            self._background_filter.config.ray_depth_enabled
            and not background_result.alignment_valid
        ):
            level = DiagnosticStatus.WARN
            message = (
                "background alignment recovering; strict RGB+3D gate active"
            )
        elif (
            self._hand_semantic_enabled
            and self._hand_semantic_status == "tracking_held"
        ):
            level = DiagnosticStatus.WARN
            message = (
                "verified hand temporarily tracked by bounded prediction"
            )
        elif (
            self._hand_semantic_enabled
            and self._hand_semantic_status
            in {"unavailable", "error", "waiting_for_rgb_projection"}
        ):
            level = DiagnosticStatus.WARN
            message = (
                "hand semantic gate unavailable; hand-only output held"
            )
        elif self_filter_result.status is SelfFilterStatus.UNAVAILABLE:
            level = DiagnosticStatus.WARN
            message = "tracking active without requested self-filter"
        elif (
            projection_error
            or geometry_fallback
            or edge_status_value == "error"
        ):
            level = DiagnosticStatus.WARN
            message = (
                "geometry tracking active with degraded optional input"
            )
        self._set_pipeline_status(
            level,
            message,
            input_mode=str(self._p("input_mode")),
            synchronized=str(bundle.synchronized),
            sync_reason=bundle.reason,
            stamp=f"{stamp:.9f}",
            input_points=str(len(sensor_cloud.points)),
            self_filtered_points=str(
                len(self_filter_result.filtered_cloud.points)
            ),
            processed_points=str(len(preprocessed.processed_cloud.points)),
            cluster_count=str(len(usable_clusters)),
            emergency_invalid_cluster_count=str(
                len(emergency_invalid_clusters)
            ),
            track_count=str(len(output_tracks)),
            hand_candidate_gate=str(
                self._intrusion_gate.config.enabled
            ).lower(),
            hand_candidate_count=str(
                len(tracks) if not geometry_safety_fallback else 0
            ),
            geometry_fallback_track_count=str(
                len(tracks) if geometry_safety_fallback else 0
            ),
            hand_semantic_enabled=str(
                self._hand_semantic_enabled
            ).lower(),
            hand_semantic_status=self._hand_semantic_status,
            hand_semantic_error=self._hand_semantic_error,
            hand_rgb_detection_count=str(hand_detection_count),
            hand_semantic_reject_reasons=hand_semantic_reject_reasons,
            prompt_count=str(len(prompts)),
            oversize_prompt_rejections=str(
                self._oversize_prompt_rejections
            ),
            preprocessing_ms=f"{preprocessing_ms:.3f}",
            clustering_ms=f"{clustering_ms:.3f}",
            tracking_ms=f"{tracking_ms:.3f}",
            fusion_ms=f"{fusion_ms:.3f}",
            projection_error=projection_error,
            edge_status=edge_status_value,
            edge_error=edge_error_value,
            self_filter_status=self_filter_result.status.value,
            self_filter_removed=str(self_filter_result.removed_points),
            plane_inliers=str(preprocessed.plane_inlier_count),
            background_state=background_result.state.value,
            background_removed=str(background_result.removed_points),
            background_baseline_points=str(
                background_result.baseline_points
            ),
            background_matched_points=str(
                background_result.matched_points
            ),
            background_alignment_points=str(
                background_result.alignment_points
            ),
            background_depth_scale=f"{background_result.depth_scale:.5f}",
            background_candidate_depth_scale=(
                f"{background_result.alignment_candidate_scale:.5f}"
            ),
            background_candidate_alignment_points=str(
                background_result.alignment_candidate_points
            ),
            background_candidate_alignment_reason=(
                background_result.alignment_candidate_reason
            ),
            background_alignment_valid=str(
                background_result.alignment_valid
            ).lower(),
            background_warmup=(
                f"{background_result.warmup_progress}/"
                f"{self._background_filter.config.warmup_frames}"
            ),
            background_calibration=(
                f"{background_result.calibration_progress}/"
                f"{self._background_filter.config.calibration_frames}"
            ),
            safety_output_state="publishing_verified",
            geometry_fallback=geometry_fallback,
            rgb_decode_error=rgb_decode_error,
        )

    def _record_processing_metrics(
        self,
        started: float,
        *,
        output_published: bool,
    ) -> None:
        """Record tracker throughput even while fail-closed output is held."""

        completed = time.monotonic()
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._state_lock:
            if output_published:
                self._last_success = completed
            self._latest_latency_ms = latency_ms
            self._recent_completion_times.append(completed)
            cutoff = completed - 2.0
            self._recent_completion_times = [
                value
                for value in self._recent_completion_times
                if value >= cutoff
            ]
            if len(self._recent_completion_times) >= 2:
                duration = (
                    self._recent_completion_times[-1]
                    - self._recent_completion_times[0]
                )
                self._latest_fps = (
                    (len(self._recent_completion_times) - 1) / duration
                    if duration > 0.0
                    else 0.0
                )

    @staticmethod
    def _normalize_rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim == 2:
            return np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("RGB input must be HxW, HxWx3, or HxWx4")
        return np.ascontiguousarray(array[..., :3])

    @staticmethod
    def _select_usable_clusters(
        clusters: Iterable[Cluster3D],
        emergency_distance: float,
    ) -> tuple[list[Cluster3D], list[Cluster3D]]:
        """Keep INVALID geometry only inside an explicitly enabled near gate."""

        candidates = list(clusters)
        emergency_enabled = float(emergency_distance) > 0.0
        emergency_invalid = [
            cluster
            for cluster in candidates
            if (
                cluster.quality is PointCloudQuality.INVALID
                and emergency_enabled
                and cluster.nearest_distance <= float(emergency_distance)
            )
        ]
        emergency_object_ids = {
            id(cluster) for cluster in emergency_invalid
        }
        usable = [
            cluster
            for cluster in candidates
            if (
                cluster.quality is not PointCloudQuality.INVALID
                or id(cluster) in emergency_object_ids
            )
        ]
        return usable, emergency_invalid

    def _camera_info_validation_error(
        self,
        camera_info: Any,
        reference_message: Any,
        expected_shape: tuple[int, int],
    ) -> str:
        """Return why a latest-only CameraInfo cannot calibrate this frame."""

        if camera_info is None or reference_message is None:
            return "CameraInfo or reference image is absent"
        height, width = (int(expected_shape[0]), int(expected_shape[1]))
        if (
            int(getattr(camera_info, "height", 0)) != height
            or int(getattr(camera_info, "width", 0)) != width
        ):
            return (
                "CameraInfo/reference shape mismatch: "
                f"{int(getattr(camera_info, 'width', 0))}x"
                f"{int(getattr(camera_info, 'height', 0))} vs "
                f"{width}x{height}"
            )
        info_header = getattr(camera_info, "header", None)
        reference_header = getattr(reference_message, "header", None)
        info_frame = str(getattr(info_header, "frame_id", ""))
        reference_frame = str(
            getattr(reference_header, "frame_id", "")
        )
        if not info_frame or not reference_frame:
            return "CameraInfo/reference frame_id is empty"
        if info_frame != reference_frame:
            return (
                "CameraInfo/reference frame mismatch: "
                f"{info_frame!r} vs {reference_frame!r}"
            )
        try:
            info_stamp = stamp_to_seconds(camera_info)
            reference_stamp = stamp_to_seconds(reference_message)
        except (TypeError, ValueError) as exc:
            return f"CameraInfo timestamp invalid: {exc}"
        allow_static = bool(self._p("sync.allow_static_camera_info"))
        if abs(info_stamp) <= 1e-12:
            if allow_static:
                return ""
            return (
                "zero-stamp CameraInfo requires "
                "sync.allow_static_camera_info=true"
            )
        slop = max(float(self._p("sync.slop_sec")), 0.0)
        delta = abs(info_stamp - reference_stamp)
        if delta > slop:
            return (
                "CameraInfo/reference timestamp mismatch: "
                f"{delta:.6f}s > {slop:.6f}s"
            )
        return ""

    def _depth_cloud_from_bundle(
        self,
        bundle: SensorBundle,
        *,
        stamp: float,
    ) -> CloudFrame:
        """Build validated depth geometry, optionally as PointCloud2 fallback."""

        if bundle.depth is None or bundle.camera_info is None:
            raise ValueError("depth or CameraInfo is absent")
        depth_stamp = stamp_to_seconds(bundle.depth)
        slop = max(float(self._p("sync.slop_sec")), 0.0)
        if abs(depth_stamp - float(stamp)) > slop:
            raise ValueError(
                "depth/geometry timestamp mismatch: "
                f"{abs(depth_stamp - float(stamp)):.6f}s > {slop:.6f}s"
            )
        depth = depth_message_to_meters(bundle.depth)
        camera_info_error = self._camera_info_validation_error(
            bundle.camera_info,
            bundle.depth,
            depth.shape,
        )
        if camera_info_error:
            raise ValueError(camera_info_error)
        intrinsics = camera_intrinsics_from_info(bundle.camera_info)
        return depth_to_cloud(
            depth,
            intrinsics,
            stamp=float(stamp),
            frame_id=str(bundle.depth.header.frame_id)
            or intrinsics.frame_id,
            # Geometry reconstruction never depends on optional RGB alignment.
            # Exact calibrated RGB projection is attached later.
            rgb=None,
            minimum_depth_m=float(
                self._p("pointcloud.minimum_depth_m")
            ),
            maximum_depth_m=float(
                self._p("pointcloud.maximum_depth_m")
            ),
        )

    @staticmethod
    def _rgb_registered_to_depth(
        rgb: np.ndarray | None,
        rgb_message: Any | None,
        depth_message: Any,
        depth_shape: tuple[int, int],
    ) -> bool:
        """Use RGB colors only for explicitly same-grid, same-frame images."""

        if rgb is None or rgb_message is None:
            return False
        if tuple(rgb.shape[:2]) != tuple(depth_shape):
            return False
        rgb_frame = str(
            getattr(getattr(rgb_message, "header", None), "frame_id", "")
        )
        depth_frame = str(
            getattr(getattr(depth_message, "header", None), "frame_id", "")
        )
        return bool(
            rgb_frame
            and depth_frame
            and rgb_frame == depth_frame
        )

    @staticmethod
    def _cloud_with_camera_pixels(
        cloud: CloudFrame,
        camera_info: Any,
        image_shape: tuple[int, int],
        tracking_to_camera: np.ndarray,
    ) -> CloudFrame:
        """Attach calibrated RGB pixels instead of assuming depth alignment."""

        projection = project_points(
            cloud.points,
            camera_info,
            image_shape,
            tracking_to_camera=tracking_to_camera,
        )
        pixels = np.full((len(cloud.points), 2), -1, dtype=np.int32)
        if len(projection.source_indices):
            pixels[projection.source_indices] = np.rint(
                projection.uv
            ).astype(np.int32)
        return replace(
            cloud,
            pixels_uv=pixels,
            image_shape=tuple(int(value) for value in image_shape),
        )

    @staticmethod
    def _cloud_inside_hand_detections(
        cloud: CloudFrame,
        detections: list[HandDetection],
        image_shape: tuple[int, int],
        *,
        tracking_to_camera: np.ndarray | None = None,
        foreground_depth_quantile: float | None = None,
        foreground_depth_span_m: float | None = None,
    ) -> CloudFrame:
        """Select the current foreground depth layer inside RGB hand masks.

        Landmark hulls bridge the gaps between fingers.  Those pixels often
        see the table behind the hand; accepting all of them creates the large
        flashing slab seen in the viewer.  Production callers provide a
        camera transform and a bounded near-depth layer so the returned points
        stay on the measured hand surface.  ``None`` preserves the legacy
        all-rays behaviour for callers that do not opt in.
        """

        height, width = map(int, image_shape)
        pixels = cloud.pixels_uv
        if pixels is None:
            return cloud.select(np.zeros(len(cloud.points), dtype=bool))
        pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        selected = np.zeros(len(cloud.points), dtype=bool)
        apply_depth_gate = (
            foreground_depth_quantile is not None
            and foreground_depth_span_m is not None
        )
        if apply_depth_gate:
            quantile = float(foreground_depth_quantile)
            span = float(foreground_depth_span_m)
            if not 0.0 <= quantile <= 1.0:
                raise ValueError("foreground_depth_quantile must be within [0, 1]")
            if not np.isfinite(span) or span <= 0.0:
                raise ValueError("foreground_depth_span_m must be positive")
            camera_points = points_in_camera_frame(
                cloud.points,
                tracking_to_camera,
            )
        for detection in detections:
            if detection.mask is None:
                continue
            mask = np.asarray(detection.mask, dtype=np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            covered = np.zeros(len(cloud.points), dtype=bool)
            covered[valid] = mask[
                pixels[valid, 1], pixels[valid, 0]
            ].astype(bool)
            if apply_depth_gate and covered.any():
                depth = camera_points[:, 2]
                measured = covered & np.isfinite(depth) & (depth > 0.0)
                if not measured.any():
                    continue
                near_depth = float(np.quantile(depth[measured], quantile))
                covered &= measured & (depth <= near_depth + span)
            selected |= covered
        return cloud.select(selected)

    @staticmethod
    def _union_clouds(first: CloudFrame, second: CloudFrame) -> CloudFrame:
        """Deduplicate a current-frame measured cloud union by source ray."""

        if first.frame_id != second.frame_id:
            raise ValueError("cannot union point clouds from different frames")
        if abs(float(first.stamp) - float(second.stamp)) > 1e-6:
            raise ValueError("cannot union point clouds from different timestamps")
        points = np.concatenate((first.points, second.points), axis=0)
        source_indices = np.concatenate(
            (first.source_indices, second.source_indices), axis=0
        )
        _, first_occurrence = np.unique(source_indices, return_index=True)
        keep = np.sort(first_occurrence)
        colors = (
            np.concatenate((first.colors, second.colors), axis=0)[keep]
            if first.colors is not None and second.colors is not None
            else None
        )
        pixels = (
            np.concatenate((first.pixels_uv, second.pixels_uv), axis=0)[keep]
            if first.pixels_uv is not None and second.pixels_uv is not None
            else None
        )
        return CloudFrame(
            points=points[keep],
            colors=colors,
            pixels_uv=pixels,
            source_indices=source_indices[keep],
            stamp=first.stamp,
            frame_id=first.frame_id,
            image_shape=first.image_shape or second.image_shape,
        )

    def _cloud_in_tracking_frame(
        self, cloud: CloudFrame, stamp: Any
    ) -> CloudFrame:
        if cloud.frame_id == self._tracking_frame:
            return cloud
        transform = self._lookup_transform(
            self._tracking_frame, cloud.frame_id, stamp
        )
        translation, quaternion = self._transform_components(transform)
        return transform_cloud(
            cloud, translation, quaternion, self._tracking_frame
        )

    def _frame_origin(self, frame_id: str, stamp: Any) -> np.ndarray:
        if not frame_id or frame_id == self._tracking_frame:
            return np.zeros(3, dtype=np.float32)
        transform = self._lookup_transform(
            self._tracking_frame, frame_id, stamp
        )
        translation, _ = self._transform_components(transform)
        return translation.astype(np.float32)

    def _link_spheres(
        self, stamp: Any
    ) -> tuple[list[LinkSphere] | None, str]:
        if not bool(self._p("self_filter.enabled")):
            return None, ""
        if str(self._p("self_filter.mode")) != "tf_spheres":
            return None, "unsupported self_filter.mode"
        frames = [str(value) for value in self._p("self_filter.link_frames")]
        radii = [
            float(value) for value in self._p("self_filter.link_radii_m")
        ]
        if not frames or len(frames) != len(radii):
            return None, "link_frames/link_radii_m are empty or mismatched"
        spheres: list[LinkSphere] = []
        try:
            for frame_id, radius in zip(frames, radii, strict=True):
                spheres.append(
                    LinkSphere(
                        center=self._frame_origin(frame_id, stamp),
                        radius=radius,
                        link_name=frame_id,
                    )
                )
        except TransformException as exc:
            return None, str(exc)
        return spheres, ""

    def _lookup_transform(
        self, target_frame: str, source_frame: str, stamp: Any
    ) -> Any:
        return self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=max(self._tf_timeout, 0.0)),
        )

    @staticmethod
    def _transform_components(
        transform_stamped: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        transform = transform_stamped.transform
        translation = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            dtype=np.float64,
        )
        return translation, quaternion

    def _tracking_to_camera_matrix(
        self, camera_info: Any, rgb_message: Any, stamp: Any
    ) -> np.ndarray:
        configured = str(self._p("frames.camera_frame"))
        camera_frame = (
            configured
            or str(getattr(camera_info.header, "frame_id", ""))
            or str(getattr(rgb_message.header, "frame_id", ""))
        )
        if camera_frame and camera_frame != self._tracking_frame:
            transform = self._lookup_transform(
                camera_frame, self._tracking_frame, stamp
            )
            translation, quaternion = self._transform_components(transform)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = quaternion_matrix_xyzw(quaternion)
            matrix[:3, 3] = translation
        else:
            matrix = np.eye(4, dtype=np.float64)
        axis_mode = str(self._p("projection.camera_axis_mode"))
        if axis_mode == "ros_optical":
            return matrix
        if axis_mode != "x_right_y_forward_z_down":
            raise ValueError(
                "projection.camera_axis_mode must be ros_optical or "
                "x_right_y_forward_z_down"
            )
        # Existing Koch points are x-right/y-forward/z-down. Pinhole
        # projection expects x-right/y-down/z-forward.
        permutation = np.eye(4, dtype=np.float64)
        permutation[:3, :3] = np.array(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        )
        return permutation @ matrix

    def _refine_cluster(
        self,
        cluster: Cluster3D,
        raw_cloud: CloudFrame,
        robot_origin: np.ndarray,
    ) -> Cluster3D:
        if len(raw_cloud.points) == 0 or len(cluster.points) == 0:
            return cluster
        padding = max(
            float(self._p("clustering.final_geometry_padding_m")), 0.0
        )
        expanded = cluster.aabb.expanded(padding)
        inside = np.all(raw_cloud.points >= expanded.minimum, axis=1)
        inside &= np.all(raw_cloud.points <= expanded.maximum, axis=1)
        candidates = np.flatnonzero(inside)
        if len(candidates) < max(
            int(self._p("clustering.min_points")), 3
        ):
            return cluster
        distances, _ = cKDTree(cluster.points).query(
            raw_cloud.points[candidates], k=1
        )
        correspondence_radius = max(
            padding,
            float(self._p("pointcloud.voxel_size")) * 1.75,
            0.005,
        )
        selected = candidates[distances <= correspondence_radius]
        if len(selected) < max(
            int(self._p("clustering.min_points")), 3
        ):
            return cluster
        points = raw_cloud.points[selected]
        aabb = AABB(np.min(points, axis=0), np.max(points, axis=0))
        obb = ClusterExtractor._pca_obb(points)
        deltas = points - robot_origin
        nearest_index = int(
            np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        )
        volume = max(aabb.volume, 1e-6)
        return Cluster3D(
            cluster_id=cluster.cluster_id,
            points=points,
            colors=(
                None
                if raw_cloud.colors is None
                else raw_cloud.colors[selected]
            ),
            source_indices=raw_cloud.source_indices[selected],
            pixels_uv=(
                None
                if raw_cloud.pixels_uv is None
                else raw_cloud.pixels_uv[selected]
            ),
            centroid=np.mean(points, axis=0),
            median_center=np.median(points, axis=0),
            aabb=aabb,
            obb=obb,
            nearest_point=points[nearest_index],
            nearest_distance=float(
                np.linalg.norm(deltas[nearest_index])
            ),
            point_count=len(points),
            density=float(len(points) / volume),
            depth_variance=float(
                np.var(points[:, self._clusterer.config.depth_axis])
            ),
            missing_depth_ratio=cluster.missing_depth_ratio,
            quality=cluster.quality,
            quality_score=cluster.quality_score,
        )

    def _make_prompts(
        self,
        tracks: list[TrackEstimate],
        camera_info: Any,
        image_shape: tuple[int, int],
        tracking_to_camera: np.ndarray,
    ) -> dict[int, ProjectionPrompt]:
        with self._edge_context_lock:
            mask_memory_ids = set(self._previous_masks)
        retained_identity_ids = (
            mask_memory_ids | set(self._hand_semantic_holds)
        )
        candidates = [
            track
            for track in tracks
            if (
                track.state is TrackingState.CONFIRMED
                or (
                    track.state
                    in {TrackingState.OCCLUDED, TrackingState.LOST}
                    and track.track_id in retained_identity_ids
                )
            )
            and len(track.source_points)
            >= int(self._p("projection.minimum_projected_points"))
        ]
        candidates.sort(key=lambda item: item.nearest_distance)
        candidates = candidates[
            : max(int(self._p("edgetam.maximum_objects")), 1)
        ]
        prompts: dict[int, ProjectionPrompt] = {}
        for track in candidates:
            reason = self._reprompt_reasons.get(track.track_id, "")
            current_stamp = float(self._last_processed_stamp or 0.0)
            interval = max(
                float(self._p("edgetam.minimum_prompt_interval_sec")),
                0.0,
            )
            last_reprompt = self._last_reprompt_stamp.get(
                track.track_id, float("-inf")
            )
            re_prompt = bool(
                reason and current_stamp - last_reprompt >= interval
            )
            prompt = project_track(
                track,
                camera_info,
                image_shape,
                frame_index=self._frame_index,
                tracking_to_camera=tracking_to_camera,
                config=self._projection_config,
                re_prompt=re_prompt,
                reason=reason or "3d_cluster_projection",
            )
            if prompt is None:
                continue
            if (
                not re_prompt
                and track.state
                in {TrackingState.OCCLUDED, TrackingState.LOST}
            ):
                prompt = replace(
                    prompt,
                    reason=(
                        f"{track.state.value.lower()}_3d_prediction"
                    ),
                )
            box = prompt.box_xyxy
            if (
                box[2] - box[0]
                < int(self._p("projection.minimum_box_width"))
                or box[3] - box[1]
                < int(self._p("projection.minimum_box_height"))
            ):
                continue
            image_area = max(int(image_shape[0]) * int(image_shape[1]), 1)
            box_area = max(float(box[2] - box[0]), 0.0) * max(
                float(box[3] - box[1]), 0.0
            )
            maximum_area_ratio = float(
                self._p("projection.maximum_box_area_ratio")
            )
            if not 0.0 < maximum_area_ratio <= 1.0:
                raise ValueError(
                    "projection.maximum_box_area_ratio must be in (0, 1]"
                )
            if box_area / image_area > maximum_area_ratio:
                self._oversize_prompt_rejections += 1
                continue
            prompts[track.track_id] = prompt
            if re_prompt:
                self._last_reprompt_stamp[track.track_id] = current_stamp
        return prompts

    def _invalidate_edge_contexts(
        self, *, reset_wrapper: bool
    ) -> None:
        """Invalidate all jobs whose geometry identity is no longer current."""

        with self._edge_context_lock:
            self._edge_context_generation += 1
            self._edge_contexts.clear()
            self._latest_masks.clear()
            self._previous_masks.clear()
            getattr(self, "_previous_mask_prompts", {}).clear()
            self._reprompt_reasons.clear()
            self._last_reprompt_stamp.clear()
            self._frames_without_cluster_points.clear()
            edge = self._edge
            if reset_wrapper and edge is not None:
                edge.reset_stream()

    def _poll_edge_result(self) -> None:
        """Non-blockingly process the latest inference result on this worker."""

        with self._edge_context_lock:
            edge = self._edge
            enabled = self._edge_enabled
        if not enabled or edge is None:
            return
        result = edge.latest_result
        if result is None:
            return
        with self._edge_context_lock:
            if not self._edge_enabled or edge is not self._edge:
                return
            if result.sequence <= self._last_edge_result_sequence:
                return
            self._last_edge_result_sequence = result.sequence
            context = self._edge_contexts.pop(result.sequence, None)
            # The wrapper is latest-only. Once N is exposed, no context older
            # than N can subsequently produce a result.
            for sequence in tuple(self._edge_contexts):
                if sequence < result.sequence:
                    self._edge_contexts.pop(sequence, None)
        try:
            self._process_edge_result(result, context)
        except Exception as exc:
            # A visual refinement failure cannot reject or delay geometry.
            error = (
                "EdgeTAM result rejected: "
                f"{type(exc).__name__}: {exc}"
            )
            self._set_edge_inference_status("error", error)
            self._latest_edge_latency_ms = result.latency_ms
            self.get_logger().warning(error)

    def _process_edge_result(
        self,
        result: EdgeTAMResult,
        context: _EdgeFrameContext | None,
    ) -> None:
        with self._edge_context_lock:
            if not self._edge_enabled:
                return
        self._latest_edge_latency_ms = result.latency_ms
        if context is None:
            with self._edge_context_lock:
                self._edge_stale_results += 1
            self._set_edge_inference_status(
                "degraded",
                f"EdgeTAM result sequence {result.sequence} has no retained "
                "exact geometry context; result discarded",
            )
            return
        with self._edge_context_lock:
            generation_current = (
                self._edge_enabled
                and context.node_generation == self._edge_context_generation
            )
        exact_identity = (
            result.frame_index == context.frame_index
            and abs(result.stamp - context.rgb_stamp) <= 1e-6
        )
        age = max(time.monotonic() - context.submitted_monotonic, 0.0)
        maximum_age = max(
            float(self._p("sync.sensor_stale_timeout_sec")), 0.0
        )
        if (
            not generation_current
            or not exact_identity
            or age > maximum_age
        ):
            with self._edge_context_lock:
                self._edge_stale_results += 1
            reason = (
                "generation invalidated"
                if not generation_current
                else (
                    "frame/stamp mismatch"
                    if not exact_identity
                    else f"context age {age:.3f}s exceeds {maximum_age:.3f}s"
                )
            )
            self._set_edge_inference_status(
                "degraded",
                f"EdgeTAM sequence {result.sequence} discarded: {reason}",
            )
            if generation_current:
                current_ids = self._matching_current_track_ids(context)
                with self._edge_context_lock:
                    if (
                        self._edge_enabled
                        and context.node_generation
                        == self._edge_context_generation
                    ):
                        for track_id in current_ids:
                            self._reprompt_reasons[track_id] = (
                                "stale_edgetam_result"
                            )
            return
        if not result.ok:
            self._set_edge_inference_status("error", result.error)
            current_ids = self._matching_current_track_ids(context)
            with self._edge_context_lock:
                if (
                    self._edge_enabled
                    and context.node_generation
                    == self._edge_context_generation
                ):
                    for track_id in current_ids & set(context.prompts):
                        self._reprompt_reasons[track_id] = (
                            "mask_disappeared_after_inference_error"
                        )
            return

        observations, quality_results = self._edge_observations(
            result, context
        )
        current_ids = self._matching_current_track_ids(context)
        self._update_edge_quality_memory(
            context, observations, quality_results, current_ids
        )
        if any(
            "independent_state" in mode
            for mode in result.prompt_modes.values()
        ):
            self._set_edge_inference_status(
                "degraded",
                "grouped multi-object propagation failed; using one "
                "official EdgeTAM state per object",
            )
        else:
            self._set_edge_inference_status("ready")

        refined_tracks = self._decorate_and_fuse_tracks(
            list(context.tracks),
            list(context.clusters),
            context.raw_cloud,
            context.prompts,
            observations,
            tracking_to_camera=context.tracking_to_camera,
            fusion_config=context.fusion_config,
        )
        has_refinement = any(
            track.edge_tam_refined and track.track_id in current_ids
            for track in refined_tracks
        )
        published = (
            self._try_publish_refined_context(
                context, refined_tracks
            )
            if has_refinement
            else False
        )
        self._publish_edge_debug_context(
            context,
            refined_tracks,
            {
                track_id: observation
                for track_id, observation in observations.items()
                if track_id in current_ids
            },
            safety_correction_published=published,
        )
        if published:
            with self._edge_context_lock:
                if (
                    self._edge_enabled
                    and context.node_generation
                    == self._edge_context_generation
                ):
                    self._edge_refined_corrections += 1
                    self._latest_masks = {
                        track_id: observation
                        for track_id, observation in observations.items()
                        if track_id in current_ids
                    }
        elif has_refinement:
            with self._edge_context_lock:
                if self._edge_enabled:
                    self._edge_stale_results += 1

    def _edge_observations(
        self,
        result: EdgeTAMResult,
        context: _EdgeFrameContext,
    ) -> tuple[
        dict[int, MaskObservation],
        dict[int, MaskQualityResult],
    ]:
        valid_depth_mask = self._valid_depth_mask(context.raw_cloud)
        with self._edge_context_lock:
            previous_masks = {
                track_id: mask.copy()
                for track_id, mask in self._previous_masks.items()
            }
        observations: dict[int, MaskObservation] = {}
        quality_results: dict[int, MaskQualityResult] = {}
        track_by_id = {
            track.track_id: track for track in context.tracks
        }
        for track_id, mask in result.masks.items():
            track = track_by_id.get(track_id)
            prompt = context.prompts.get(track_id)
            if track is None or prompt is None:
                continue
            observation = MaskObservation(
                track_id=track_id,
                frame_index=result.frame_index,
                stamp=result.stamp,
                mask=mask,
            )
            previous_mask = previous_masks.get(track_id)
            if (
                previous_mask is not None
                and previous_mask.shape != observation.mask.shape
            ):
                previous_mask = None
                observation.re_prompted = True
                with self._edge_context_lock:
                    if (
                        self._edge_enabled
                        and context.node_generation
                        == self._edge_context_generation
                    ):
                        self._previous_masks.pop(track_id, None)
                        getattr(
                            self, "_previous_mask_prompts", {}
                        ).pop(track_id, None)
                        self._reprompt_reasons[track_id] = (
                            "rgb_resolution_changed"
                        )
            measured_centroid = self._mask_depth_centroid(
                context.raw_cloud, observation.mask
            )
            prediction_kwargs: dict[str, np.ndarray] = {}
            if measured_centroid is not None:
                prediction_kwargs = {
                    "predicted_centroid": track.position,
                    "measured_centroid": measured_centroid,
                }
            quality = apply_mask_quality(
                observation,
                prompt.projection_mask,
                valid_depth_mask=valid_depth_mask,
                previous_mask=previous_mask,
                config=self._mask_quality_config,
                **prediction_kwargs,
            )
            observations[track_id] = observation
            quality_results[track_id] = quality
        quality_counts = {
            quality: sum(
                observation.quality is quality
                for observation in observations.values()
            )
            for quality in (
                MaskQuality.GOOD,
                MaskQuality.DEGRADED,
                MaskQuality.INVALID,
            )
        }
        reject_reasons = sorted(
            {
                reason
                for quality in quality_results.values()
                for reason in quality.reasons
            }
        )
        with self._edge_context_lock:
            if (
                self._edge_enabled
                and context.node_generation == self._edge_context_generation
            ):
                self._latest_mask_diagnostics = {
                    "mask_count": str(len(observations)),
                    "mask_good_count": str(
                        quality_counts[MaskQuality.GOOD]
                    ),
                    "mask_degraded_count": str(
                        quality_counts[MaskQuality.DEGRADED]
                    ),
                    "mask_invalid_count": str(
                        quality_counts[MaskQuality.INVALID]
                    ),
                    "mask_reject_reasons": ",".join(reject_reasons),
                }
        return observations, quality_results

    def _matching_current_track_ids(
        self, context: _EdgeFrameContext
    ) -> set[int]:
        """Match persistent identity, including protection against ID reuse."""

        timestamp = float(
            self._last_processed_stamp
            if self._last_processed_stamp is not None
            else context.geometry_stamp
        )
        current = {
            track.track_id: track
            for track in self._tracker.predict_to(timestamp)
            if track.state is not TrackingState.DELETED
        }
        context_by_id = {
            track.track_id: track for track in context.tracks
        }
        return {
            track_id
            for track_id, track in current.items()
            if (
                track_id in context_by_id
                and abs(
                    track.first_timestamp
                    - context_by_id[track_id].first_timestamp
                )
                <= 1e-9
            )
        }

    def _update_edge_quality_memory(
        self,
        context: _EdgeFrameContext,
        observations: dict[int, MaskObservation],
        quality_results: dict[int, MaskQualityResult],
        current_ids: set[int],
    ) -> None:
        track_by_id = {
            track.track_id: track for track in context.tracks
        }
        # TrackEstimate already carries the exact one-to-one Hungarian
        # measurement timestamp. Cached source points on an occluded track do
        # not count as a current cluster measurement.
        cluster_present_by_id = {
            track_id: (
                track.state
                in {
                    TrackingState.TENTATIVE,
                    TrackingState.CONFIRMED,
                }
                and abs(
                    track.last_measurement_timestamp
                    - context.geometry_stamp
                )
                <= 1e-6
            )
            for track_id, track in track_by_id.items()
        }
        with self._edge_context_lock:
            if (
                not self._edge_enabled
                or context.node_generation
                != self._edge_context_generation
            ):
                return
            for track_id, observation in observations.items():
                if track_id not in current_ids:
                    continue
                prompt = context.prompts[track_id]
                projection_mask = prompt.projection_mask
                cluster_overlap = (
                    0
                    if projection_mask is None
                    else int(
                        np.count_nonzero(
                            observation.mask & projection_mask
                        )
                    )
                )
                if cluster_overlap == 0:
                    self._frames_without_cluster_points[track_id] = (
                        self._frames_without_cluster_points.get(track_id, 0)
                        + 1
                    )
                else:
                    self._frames_without_cluster_points[track_id] = 0
                previous_present = track_id in self._previous_masks
                decision = decide_reprompt(
                    quality_results[track_id],
                    cluster_present=cluster_present_by_id.get(
                        track_id, False
                    ),
                    previous_mask_present=previous_present,
                    frames_without_cluster_points=(
                        self._frames_without_cluster_points.get(track_id, 0)
                    ),
                    reappeared_after_occlusion=(
                        prompt.re_prompt
                        and prompt.reason == "reappeared_after_occlusion"
                    ),
                    config=self._reprompt_config,
                )
                if observation.re_prompted:
                    self._reprompt_reasons[track_id] = (
                        "rgb_resolution_changed"
                    )
                    self._last_reprompt_stamp.pop(track_id, None)
                elif decision.required:
                    self._reprompt_reasons[track_id] = ",".join(
                        decision.reasons
                    )
                else:
                    self._reprompt_reasons.pop(track_id, None)
                self._previous_masks[track_id] = (
                    observation.mask.copy()
                )
                if not hasattr(self, "_previous_mask_prompts"):
                    self._previous_mask_prompts = {}
                self._previous_mask_prompts[track_id] = deepcopy(prompt)
            for track_id in tuple(self._previous_masks):
                if track_id not in current_ids:
                    self._previous_masks.pop(track_id, None)
                    getattr(
                        self, "_previous_mask_prompts", {}
                    ).pop(track_id, None)
                    self._reprompt_reasons.pop(track_id, None)
                    self._last_reprompt_stamp.pop(track_id, None)
                    self._frames_without_cluster_points.pop(
                        track_id, None
                    )

    def _try_publish_refined_context(
        self,
        context: _EdgeFrameContext,
        tracks: list[TrackEstimate],
    ) -> bool:
        """Atomically reject a correction after any newer safety output."""

        age = max(time.monotonic() - context.submitted_monotonic, 0.0)
        maximum_age = max(
            float(self._p("sync.sensor_stale_timeout_sec")), 0.0
        )
        with self._edge_context_lock:
            if (
                not self._edge_enabled
                or context.node_generation
                != self._edge_context_generation
                or age > maximum_age
            ):
                return False
            with self._publication_lock:
                if (
                    self._safety_publication_serial
                    != context.publication_serial
                    or self._last_safety_output_stamp is None
                    or abs(
                        self._last_safety_output_stamp
                        - context.geometry_stamp
                    )
                    > 1e-9
                ):
                    return False
                self._publish_safety_outputs(context.header, tracks)
                self._safety_publication_serial += 1
                # The correction deliberately retains the exact geometry stamp.
                self._last_safety_output_stamp = context.geometry_stamp
                return True

    def _publish_edge_debug_context(
        self,
        context: _EdgeFrameContext,
        tracks: list[TrackEstimate],
        observations: dict[int, MaskObservation],
        *,
        safety_correction_published: bool,
    ) -> None:
        """Publish a timestamp-honest overlay for the exact RGB inference."""

        publisher = self._debug_image_publisher
        if publisher is None or not observations:
            return
        age = max(time.monotonic() - context.submitted_monotonic, 0.0)
        maximum_age = max(
            float(self._p("sync.sensor_stale_timeout_sec")), 0.0
        )
        with self._edge_context_lock:
            if (
                not self._edge_enabled
                or context.node_generation
                != self._edge_context_generation
                or age > maximum_age
            ):
                return
        debug_image = render_debug_image(
            context.rgb,
            tracks,
            context.prompts,
            observations,
            status_text=(
                f"MASK frame={context.frame_index} · "
                f"latency={self._latest_edge_latency_ms:.1f}ms · "
                "safety_correction="
                f"{'published' if safety_correction_published else 'cached'}"
            ),
        )
        publisher.publish(
            make_rgb_image_message(
                debug_image,
                header=context.rgb_header,
                image_type=Image,
            )
        )

    def _publish_live_debug_snapshot(
        self,
        rgb: np.ndarray | None,
        rgb_header: Any | None,
        *,
        status_text: str,
    ) -> None:
        """Keep the GUI current when no trustworthy EdgeTAM prompt exists."""

        publisher = self._debug_image_publisher
        if publisher is None or rgb is None or rgb_header is None:
            return
        debug_image = render_debug_image(
            rgb,
            [],
            {},
            {},
            status_text=status_text,
        )
        publisher.publish(
            make_rgb_image_message(
                debug_image,
                header=rgb_header,
                image_type=Image,
            )
        )

    @staticmethod
    def _valid_depth_mask(cloud: CloudFrame) -> np.ndarray | None:
        if cloud.image_shape is None or cloud.pixels_uv is None:
            return None
        height, width = cloud.image_shape
        result = np.zeros((height, width), dtype=bool)
        pixels = cloud.pixels_uv
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
            & np.isfinite(cloud.points).all(axis=1)
        )
        result[pixels[valid, 1], pixels[valid, 0]] = True
        return result

    @staticmethod
    def _mask_depth_centroid(
        cloud: CloudFrame, mask: np.ndarray
    ) -> np.ndarray | None:
        if cloud.image_shape is None or cloud.pixels_uv is None:
            return None
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != cloud.image_shape:
            return None
        height, width = cloud.image_shape
        pixels = cloud.pixels_uv
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
            & np.isfinite(cloud.points).all(axis=1)
        )
        selected = np.zeros(len(cloud.points), dtype=bool)
        selected[valid] = candidate[
            pixels[valid, 1], pixels[valid, 0]
        ]
        if not np.any(selected):
            return None
        return np.median(cloud.points[selected], axis=0).astype(np.float32)

    def _decorate_and_fuse_tracks(
        self,
        tracks: list[TrackEstimate],
        clusters: list[Cluster3D],
        raw_cloud: CloudFrame,
        prompts: dict[int, ProjectionPrompt],
        masks: dict[int, MaskObservation],
        *,
        tracking_to_camera: np.ndarray | None,
        fusion_config: FusionConfig | None = None,
    ) -> list[TrackEstimate]:
        result: list[TrackEstimate] = []
        exact_fusion_config = (
            self._fusion_config
            if fusion_config is None
            else fusion_config
        )
        cloud_camera_depths = (
            None
            if tracking_to_camera is None
            else points_in_camera_frame(
                raw_cloud.points,
                tracking_to_camera,
            )[:, 2]
        )
        for track in tracks:
            observation = masks.get(track.track_id)
            prompt = prompts.get(track.track_id)
            # Never redo the tracker's global one-to-one association with a
            # local centroid-nearest lookup. TrackEstimate geometry is the
            # exact matched measurement (or its fail-safe cached prediction).
            cluster = (
                self._track_geometry_proxy(
                    track, current_cloud_stamp=raw_cloud.stamp
                )
                if len(track.source_points)
                else None
            )
            using_predicted_proxy = (
                track.state
                in {TrackingState.OCCLUDED, TrackingState.LOST}
                and observation is not None
                and prompt is not None
            )
            decorated = track
            if observation is not None:
                decorated = replace(
                    decorated,
                    mask_quality=observation.quality,
                    mask_quality_score=observation.quality_score,
                )
            if (
                observation is not None
                and prompt is not None
                and cluster is not None
                and observation.quality
                in {MaskQuality.GOOD, MaskQuality.DEGRADED}
            ):
                fused = fuse_mask_with_cloud(
                    observation,
                    raw_cloud,
                    cluster,
                    predicted_geometry=track,
                    projection_mask=prompt.projection_mask,
                    config=exact_fusion_config,
                    point_depths=cloud_camera_depths,
                    cluster_depths=(
                        None
                        if tracking_to_camera is None
                        else points_in_camera_frame(
                            cluster.points,
                            tracking_to_camera,
                        )[:, 2]
                    ),
                )
                if fused.used_mask and len(fused.points):
                    decorated = replace(
                        decorated,
                        source_points=fused.points,
                        source_colors=fused.colors,
                        source_indices=fused.source_indices,
                        aabb=fused.aabb,
                        obb=fused.obb,
                        nearest_point=fused.nearest_point,
                        nearest_distance=fused.nearest_distance,
                        point_count=fused.fused_point_count,
                        edge_tam_refined=True,
                    )
            temporal_score = min(
                track.hit_count
                / max(float(self._tracker_config.confirmation_hits), 1.0),
                1.0,
            )
            mask_score: float | None
            if observation is None or observation.quality is MaskQuality.UNAVAILABLE:
                mask_score = None
            else:
                mask_score = (
                    observation.quality_score
                    if observation.quality is not MaskQuality.INVALID
                    else 0.0
                )
            confidence = compute_fused_confidence(
                track.confidence,
                temporal_score,
                mask_score,
                nearest_distance=decorated.nearest_distance,
                config=self._confidence_config,
            )
            confidence_score = confidence.score
            uncertainty_margin = confidence.uncertainty_margin
            if using_predicted_proxy:
                # RGB can recover aligned depth points but cannot turn a
                # missing 3D measurement into a fresh point-cloud observation.
                confidence_score = min(
                    confidence_score, float(track.confidence)
                )
                uncertainty_margin = max(
                    uncertainty_margin,
                    float(self._p("safety.base_uncertainty_margin_m"))
                    + 0.02 * max(track.missed_count, 1),
                )
            decorated = replace(
                decorated,
                confidence=confidence_score,
                uncertainty_margin=min(
                    max(
                        decorated.uncertainty_margin,
                        uncertainty_margin,
                    ),
                    float(
                        self._p("safety.maximum_uncertainty_margin_m")
                    ),
                ),
            )
            result.append(decorated)
        return result

    def _track_geometry_proxy(
        self,
        track: TrackEstimate,
        *,
        current_cloud_stamp: float | None = None,
    ) -> Cluster3D:
        """Preserve the tracker's exact one-to-one measurement as fusion base."""

        points = np.asarray(track.source_points, dtype=np.float32)
        depth_variance = (
            float(
                np.var(
                    points[:, self._clusterer.config.depth_axis]
                )
            )
            if len(points)
            else 0.0
        )
        indices_are_same_frame = (
            current_cloud_stamp is None
            or abs(
                track.last_measurement_timestamp
                - float(current_cloud_stamp)
            )
            <= 1e-6
        )
        return Cluster3D(
            cluster_id=-int(track.track_id),
            points=points,
            colors=track.source_colors,
            # PointCloud source indices are frame-local. Reusing old IDs for an
            # OCCLUDED prediction could deduplicate away a closer current depth
            # point that happens to have the same numeric pixel/record index.
            source_indices=(
                track.source_indices if indices_are_same_frame else None
            ),
            centroid=track.position,
            median_center=(
                np.median(points, axis=0)
                if len(points)
                else track.position
            ),
            aabb=track.aabb,
            obb=track.obb,
            nearest_point=track.nearest_point,
            nearest_distance=track.nearest_distance,
            point_count=len(points),
            density=(
                float(len(points) / max(track.aabb.volume, 1e-6))
                if len(points)
                else 0.0
            ),
            depth_variance=depth_variance,
            missing_depth_ratio=(
                1.0
                if track.pointcloud_quality is PointCloudQuality.INVALID
                else 0.0
            ),
            quality=track.pointcloud_quality,
            quality_score=(
                0.0
                if track.pointcloud_quality is PointCloudQuality.INVALID
                else float(track.confidence)
            ),
        )

    def _submit_edgetam(
        self,
        rgb: np.ndarray,
        prompts: list[ProjectionPrompt],
        *,
        rgb_stamp: float,
        geometry_stamp: float,
        header: Header,
        rgb_header: Header,
        tracks: list[TrackEstimate],
        clusters: list[Cluster3D],
        raw_cloud: CloudFrame,
        prompts_by_id: dict[int, ProjectionPrompt],
        tracking_to_camera: np.ndarray | None,
        publication_serial: int,
    ) -> int | None:
        with self._edge_context_lock:
            edge = self._edge
            enabled = self._edge_enabled
        if not enabled or edge is None or not edge.available or not prompts:
            return None
        try:
            edge_prompts = prompts
            if not bool(
                self._p("projection.use_projection_mask_prompt")
            ):
                edge_prompts = [
                    replace(prompt, projection_mask=None)
                    for prompt in prompts
                ]
            # Snapshot outside the wrapper's queue lock. All arrays are copied
            # so a later tracker update cannot change this job's 3D identity.
            context_tracks = tuple(deepcopy(tracks))
            context_clusters = tuple(deepcopy(clusters))
            context_cloud = deepcopy(raw_cloud)
            context_prompts = deepcopy(prompts_by_id)
            context_transform = (
                None
                if tracking_to_camera is None
                else np.asarray(
                    tracking_to_camera, dtype=np.float64
                ).copy()
            )
            context_header = deepcopy(header)
            context_rgb_header = deepcopy(rgb_header)
            context_rgb = np.asarray(rgb, dtype=np.uint8).copy()
            context_fusion_config = replace(self._fusion_config)
            with self._edge_context_lock:
                if not self._edge_enabled or edge is not self._edge:
                    return None
                node_generation = self._edge_context_generation
                sequence = edge.submit(
                    rgb,
                    self._frame_index,
                    float(rgb_stamp),
                    edge_prompts,
                    active_track_ids=[
                        prompt.track_id for prompt in prompts
                    ],
                )
                self._edge_contexts[sequence] = _EdgeFrameContext(
                    sequence=sequence,
                    node_generation=node_generation,
                    frame_index=self._frame_index,
                    geometry_stamp=float(geometry_stamp),
                    rgb_stamp=float(rgb_stamp),
                    submitted_monotonic=time.monotonic(),
                    publication_serial=int(publication_serial),
                    header=context_header,
                    rgb_header=context_rgb_header,
                    rgb=context_rgb,
                    tracks=context_tracks,
                    clusters=context_clusters,
                    raw_cloud=context_cloud,
                    prompts=context_prompts,
                    tracking_to_camera=context_transform,
                    fusion_config=context_fusion_config,
                )
                while len(self._edge_contexts) > self._edge_context_limit:
                    self._edge_contexts.popitem(last=False)
            return sequence
        except EdgeTAMError as exc:
            self._set_edge_inference_status("error", str(exc))
            return None

    def _live_predicted_masks(
        self,
        prompts: dict[int, ProjectionPrompt],
        *,
        stamp: float,
    ) -> dict[int, MaskObservation]:
        """Move the last exact EdgeTAM masks to current 3D prompt boxes.

        The official finite-video API is asynchronous and slower than the 3D
        tracker. This affine prediction is display-only on the immediate path;
        exact mask/cloud fusion still occurs only in ``_process_edge_result``
        against the saved same-stamp context.
        """

        if not prompts:
            return {}
        with self._edge_context_lock:
            previous_masks = {
                track_id: mask.copy()
                for track_id, mask in self._previous_masks.items()
            }
            previous_prompts = {
                track_id: deepcopy(prompt)
                for track_id, prompt
                in getattr(
                    self, "_previous_mask_prompts", {}
                ).items()
            }
        predicted: dict[int, MaskObservation] = {}
        for track_id, current_prompt in prompts.items():
            mask = previous_masks.get(track_id)
            previous_prompt = previous_prompts.get(track_id)
            if mask is None or previous_prompt is None:
                continue
            expected_shape = (
                None
                if current_prompt.projection_mask is None
                else current_prompt.projection_mask.shape
            )
            if expected_shape is not None and mask.shape != expected_shape:
                continue
            previous_box = np.asarray(
                previous_prompt.box_xyxy, dtype=np.float32
            )
            current_box = np.asarray(
                current_prompt.box_xyxy, dtype=np.float32
            )
            previous_size = previous_box[2:] - previous_box[:2]
            current_size = current_box[2:] - current_box[:2]
            if (
                not np.isfinite(previous_box).all()
                or not np.isfinite(current_box).all()
                or np.any(previous_size <= 1.0)
                or np.any(current_size <= 1.0)
            ):
                continue
            scale = current_size / previous_size
            if np.any(scale < 0.35) or np.any(scale > 2.85):
                continue
            offset = current_box[:2] - scale * previous_box[:2]
            transform = np.asarray(
                (
                    (float(scale[0]), 0.0, float(offset[0])),
                    (0.0, float(scale[1]), float(offset[1])),
                ),
                dtype=np.float32,
            )
            height, width = mask.shape
            warped = cv2.warpAffine(
                mask.astype(np.uint8),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            if not warped.any():
                continue
            predicted[track_id] = MaskObservation(
                track_id=track_id,
                frame_index=self._frame_index,
                stamp=float(stamp),
                mask=warped,
                quality=MaskQuality.DEGRADED,
                quality_score=0.40,
            )
        return predicted

    def _publish_outputs(
        self,
        header: Header,
        tracks: list[TrackEstimate],
        debug_cloud: CloudFrame,
        self_filtered_cloud: CloudFrame,
        rgb: np.ndarray | None,
        rgb_header: Header | None,
        prompts: dict[int, ProjectionPrompt],
        masks: dict[int, MaskObservation],
        hand_detections: list[HandDetection] | None = None,
    ) -> int | None:
        published = self._publish_new_safety_output(
            header, tracks
        )
        if published is None:
            return None
        obstacle_points, obstacle_colors, publication_serial = published
        if self._debug_cloud_publisher is not None:
            # Debug cloud is deliberately colored by persistent Track ID.
            self._debug_cloud_publisher.publish(
                make_pointcloud2(
                    obstacle_points,
                    obstacle_colors,
                    header=header,
                    pointcloud_type=PointCloud2,
                    pointfield_type=PointField,
                )
            )
        if self._self_filtered_publisher is not None:
            self._self_filtered_publisher.publish(
                make_pointcloud2(
                    self_filtered_cloud.points,
                    self_filtered_cloud.colors,
                    header=header,
                    pointcloud_type=PointCloud2,
                    pointfield_type=PointField,
                )
            )
        if rgb is not None and self._debug_image_publisher is not None:
            with self._edge_context_lock:
                edge_status_value = self._edge_status
                edge_enabled = self._edge_enabled
            predicted_masks = (
                {}
                if masks or not edge_enabled
                else self._live_predicted_masks(
                    prompts,
                    stamp=stamp_to_seconds(
                        header if rgb_header is None else rgb_header
                    ),
                )
            )
            display_masks = masks or predicted_masks
            debug_image = render_debug_image(
                rgb,
                tracks,
                prompts,
                display_masks,
                status_text=(
                    f"LIVE frame={self._frame_index} · 3D tracker · "
                    f"EdgeTAM={edge_status_value} · "
                    f"predicted_masks={len(predicted_masks)}"
                ),
                hand_detections=hand_detections or (),
            )
            self._debug_image_publisher.publish(
                make_rgb_image_message(
                    debug_image,
                    header=(header if rgb_header is None else rgb_header),
                    image_type=Image,
                )
            )
        return publication_serial

    def _publish_new_safety_output(
        self,
        header: Header,
        tracks: list[TrackEstimate],
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Publish a new output only when its timestamp cannot regress."""

        output_stamp = stamp_to_seconds(header)
        with self._publication_lock:
            if (
                self._last_safety_output_stamp is not None
                and output_stamp < self._last_safety_output_stamp - 1e-9
            ):
                return None
            points, colors = self._publish_safety_outputs(header, tracks)
            self._safety_publication_serial += 1
            self._last_safety_output_stamp = output_stamp
            return (
                points,
                colors,
                self._safety_publication_serial,
            )

    def _reset_safety_publication_epoch(self) -> None:
        """Allow a confirmed ROS/bag clock reset to start a new stamp epoch."""

        with self._publication_lock:
            # Keep the serial globally monotonic so no pre-reset correction can
            # accidentally satisfy a reused value.
            self._safety_publication_serial += 1
            self._last_safety_output_stamp = None
        self._last_prediction_publish = 0.0

    def _publish_safety_outputs(
        self,
        header: Header,
        tracks: list[TrackEstimate],
    ) -> tuple[np.ndarray, np.ndarray]:
        obstacle_array = make_tracked_obstacle_array(
            tracks,
            header=header,
            obstacle_type=TrackedObstacle,
            array_type=TrackedObstacleArray,
            point_type=Point,
        )
        self._obstacles_publisher.publish(obstacle_array)
        obstacle_points, obstacle_colors = _fresh_measured_obstacle_cloud(
            tracks,
            stamp_to_seconds(header),
        )
        obstacle_cloud = make_pointcloud2(
            obstacle_points,
            obstacle_colors,
            header=header,
            pointcloud_type=PointCloud2,
            pointfield_type=PointField,
        )
        self._cloud_publisher.publish(obstacle_cloud)
        if self._legacy_cloud_publisher is not None:
            self._legacy_cloud_publisher.publish(obstacle_cloud)
        if self._markers_publisher is not None:
            self._markers_publisher.publish(
                make_marker_array(
                    tracks,
                    header=header,
                    marker_type=Marker,
                    marker_array_type=MarkerArray,
                    point_type=Point,
                )
            )
        return obstacle_points, obstacle_colors

    def _maybe_publish_prediction(self) -> None:
        with self._state_lock:
            last_success = self._last_success
        if last_success <= 0.0:
            return
        monotonic_now = time.monotonic()
        measurement_gap = monotonic_now - last_success
        stale_timeout = max(
            float(self._p("sync.sensor_stale_timeout_sec")), 0.0
        )
        start_delay = max(
            2.0
            * float(self._p("sync.pointcloud_fallback_delay_sec")),
            0.05,
        )
        rate = max(
            float(self._p("performance.prediction_publish_rate_hz")),
            0.1,
        )
        publish_period = 1.0 / rate
        if (
            measurement_gap < start_delay
            or measurement_gap >= stale_timeout
            or monotonic_now - self._last_prediction_publish
            < publish_period
        ):
            return
        prediction_time = self.get_clock().now()
        prediction_seconds = prediction_time.nanoseconds * 1e-9
        if (
            self._last_processed_stamp is None
            or prediction_seconds <= self._last_processed_stamp
        ):
            return
        tracks = [
            track
            for track in self._tracker.predict_to(prediction_seconds)
            if track.state is not TrackingState.DELETED
            and track.track_id in self._intrusion_gate.accepted_track_ids
            and (
                not self._hand_semantic_enabled
                or track.track_id in self._hand_semantic_holds
            )
        ]
        predicted_tracks: list[TrackEstimate] = []
        for track in tracks:
            age = max(
                prediction_seconds - track.last_measurement_timestamp,
                0.0,
            )
            if track.state in {
                TrackingState.CONFIRMED,
                TrackingState.TENTATIVE,
            }:
                predicted_state = (
                    TrackingState.OCCLUDED
                    if track.state is TrackingState.CONFIRMED
                    else TrackingState.LOST
                )
            else:
                predicted_state = track.state
            confidence_decay = float(
                np.exp(
                    -age
                    / max(
                        self._tracker_config.lost_retention_seconds,
                        1e-3,
                    )
                )
            )
            predicted_tracks.append(
                replace(
                    track,
                    state=predicted_state,
                    confidence=float(
                        np.clip(
                            track.confidence * confidence_decay,
                            0.0,
                            1.0,
                        )
                    ),
                    mask_quality=MaskQuality.UNAVAILABLE,
                    mask_quality_score=0.0,
                    pointcloud_quality=PointCloudQuality.INVALID,
                    edge_tam_refined=False,
                    semantic_class=(
                        "hand"
                        if self._hand_semantic_enabled
                        else "hand_candidate"
                    ),
                    semantic_confirmed=True,
                    uncertainty_margin=min(
                        max(
                            track.uncertainty_margin,
                            float(
                                self._p(
                                    "safety.base_uncertainty_margin_m"
                                )
                            )
                            + age * 0.1,
                        ),
                        float(
                            self._p(
                                "safety.maximum_uncertainty_margin_m"
                            )
                        ),
                    ),
                )
            )
        with self._state_lock:
            entered_geometry_gap = not self._geometry_gap_active
            self._geometry_gap_active = True
            hard_error_active = (
                self._pipeline_level == DiagnosticStatus.ERROR
            )
        if entered_geometry_gap:
            self._invalidate_edge_contexts(reset_wrapper=True)
        if not predicted_tracks:
            if not hard_error_active:
                self._set_pipeline_status(
                    DiagnosticStatus.WARN,
                    "geometry temporarily missing; no prior track to predict",
                    prediction_only="true",
                    measurement_gap_sec=f"{measurement_gap:.3f}",
                    predicted_track_count="0",
                )
            return
        header = Header()
        header.stamp = prediction_time.to_msg()
        header.frame_id = self._tracking_frame
        if self._publish_new_safety_output(
            header, predicted_tracks
        ) is None:
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "non-monotonic prediction publication rejected",
                prediction_stamp=f"{prediction_seconds:.9f}",
            )
            return
        self._last_prediction_publish = monotonic_now
        if not hard_error_active:
            self._set_pipeline_status(
                DiagnosticStatus.WARN,
                "geometry temporarily missing; publishing bounded Kalman prediction",
                prediction_only="true",
                measurement_gap_sec=f"{measurement_gap:.3f}",
                predicted_track_count=str(len(predicted_tracks)),
            )

    def _load_edgetam(self) -> None:
        edge: EdgeTAMWrapper | None = None
        try:
            input_width = int(self._p("edgetam.input_width"))
            input_height = int(self._p("edgetam.input_height"))
            if (input_width, input_height) != (1024, 1024):
                raise ValueError(
                    "Pinned official EdgeTAM config requires 1024x1024 "
                    "model input"
                )
            edge = EdgeTAMWrapper(
                EdgeTAMConfig(
                    repository_path=self._resolve_path(
                        str(self._p("edgetam.repository_path"))
                    ),
                    checkpoint_path=self._resolve_path(
                        str(self._p("edgetam.checkpoint_path"))
                    ),
                    model_config=str(self._p("edgetam.model_config")),
                    device=str(self._p("edgetam.device")),
                    precision=str(self._p("edgetam.precision")),
                    window_size=int(
                        self._p("edgetam.rolling_window_frames")
                    ),
                    jpeg_quality=int(self._p("edgetam.jpeg_quality")),
                    offload_video_to_cpu=bool(
                        self._p("edgetam.offload_video_to_cpu")
                    ),
                    offload_state_to_cpu=bool(
                        self._p("edgetam.offload_state_to_cpu")
                    ),
                    clear_memory_on_reset=bool(
                        self._p("edgetam.clear_memory_on_reset")
                    ),
                )
            )
            edge.load(start_worker=True)
            with self._edge_context_lock:
                shutting_down = self._shutdown.is_set()
                if not shutting_down:
                    self._edge = edge
                    enabled = self._edge_enabled
                    self._edge_status = "ready" if enabled else "disabled"
                    self._edge_error = ""
                else:
                    enabled = False
            if shutting_down:
                edge.close(timeout=2.0)
                return
            self.get_logger().info(
                "EdgeTAM loaded on "
                f"{edge.resolved_device} ({edge.resolved_precision}); "
                f"refinement={'enabled' if enabled else 'disabled'}"
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._edge_context_lock:
                if self._edge_enabled:
                    self._edge_status = "error"
                    self._edge_error = error
                else:
                    self._edge_status = "disabled"
                    self._edge_error = ""
            self.get_logger().warning(
                "EdgeTAM unavailable; continuing point-cloud-only: "
                f"{error}"
            )

    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        source_candidate = (
            Path(__file__).resolve().parents[2] / path
        ).resolve()
        return (
            source_candidate
            if source_candidate.exists()
            else cwd_candidate
        )

    def _set_pipeline_status(
        self, level: int, message: str, **values: str
    ) -> None:
        with self._state_lock:
            # ROS 2 Humble generates uint8 constants as one-byte values while
            # newer generators may expose Python ints. Preserve either form;
            # the generated message setter accepts its native constant type.
            self._pipeline_level = level
            self._pipeline_message = message
            self._diagnostic_values = {
                key: str(value) for key, value in values.items()
            }

    def _check_stale(self) -> None:
        with self._state_lock:
            last_received = self._last_geometry_received
            last_success = self._last_success
        now = time.monotonic()
        timeout = max(
            float(self._p("sync.sensor_stale_timeout_sec")), 0.0
        )
        if last_received == 0.0:
            self._set_pipeline_status(
                DiagnosticStatus.WARN, "waiting for first geometry frame"
            )
        elif now - last_received > timeout:
            # Deliberately publish neither an empty scene nor stale points. A
            # safety consumer must treat this diagnostic/output timeout as stop.
            with self._state_lock:
                self._latest_fps = 0.0
                entered_geometry_gap = not self._geometry_gap_active
                self._geometry_gap_active = True
            if entered_geometry_gap:
                self._invalidate_edge_contexts(reset_wrapper=True)
            self._set_pipeline_status(
                DiagnosticStatus.ERROR,
                "geometry stream stale; safety output publication stopped",
                geometry_age_sec=f"{now - last_received:.3f}",
                last_success_age_sec=(
                    "never"
                    if last_success == 0.0
                    else f"{now - last_success:.3f}"
                ),
            )

    def _publish_diagnostics(self) -> None:
        with self._edge_context_lock:
            edge_refined_corrections = self._edge_refined_corrections
            edge_stale_results = self._edge_stale_results
            retained_edge_contexts = len(self._edge_contexts)
            edge_enabled = self._edge_enabled
            edge_state = self._edge_status
            edge_error = self._edge_error
            mask_diagnostics = dict(
                getattr(
                    self,
                    "_latest_mask_diagnostics",
                    {
                        "mask_count": "0",
                        "mask_good_count": "0",
                        "mask_degraded_count": "0",
                        "mask_invalid_count": "0",
                        "mask_reject_reasons": "",
                    },
                )
            )
        with self._state_lock:
            level = self._pipeline_level
            message = self._pipeline_message
            values = dict(self._diagnostic_values)
            values.update(
                {
                    "dropped_geometry_bundles": str(
                        self._dropped_bundles
                    ),
                    "fps": f"{self._latest_fps:.3f}",
                    "latency_ms": f"{self._latest_latency_ms:.3f}",
                    "edge_latency_ms": (
                        f"{self._latest_edge_latency_ms:.3f}"
                    ),
                    "edge_enabled": str(edge_enabled).lower(),
                    "edge_status": edge_state,
                    "edge_error": edge_error,
                    "edge_refined_corrections": str(
                        edge_refined_corrections
                    ),
                    "edge_stale_results": str(edge_stale_results),
                    "retained_edge_contexts": str(
                        retained_edge_contexts
                    ),
                    **mask_diagnostics,
                }
            )
            fps = self._latest_fps
            latency = self._latest_latency_ms
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        diagnostic = DiagnosticArray()
        diagnostic.header = header
        status = DiagnosticStatus()
        status.level = level
        status.name = f"{self.get_fully_qualified_name()}/pipeline"
        status.hardware_id = "rgbd_pointcloud_tracker"
        status.message = message
        status.values = [
            KeyValue(key=key, value=value)
            for key, value in sorted(values.items())
        ]
        edge_status = DiagnosticStatus()
        edge_status.name = (
            f"{self.get_fully_qualified_name()}/edgetam"
        )
        edge_status.hardware_id = "official_edgetam"
        if edge_state == "disabled":
            edge_status.level = DiagnosticStatus.OK
            edge_status.message = (
                "EdgeTAM refinement disabled; point-cloud tracking remains active"
            )
        elif edge_state == "error":
            edge_status.level = DiagnosticStatus.ERROR
            edge_status.message = (
                "EdgeTAM unavailable/failed; point-cloud fallback remains active"
            )
        elif edge_state == "loading":
            edge_status.level = DiagnosticStatus.WARN
            edge_status.message = "EdgeTAM model is loading"
        elif edge_state == "degraded":
            edge_status.level = DiagnosticStatus.WARN
            edge_status.message = (
                "EdgeTAM compatibility fallback active; point-cloud "
                "safety path remains authoritative"
            )
        else:
            edge_status.level = DiagnosticStatus.OK
            edge_status.message = edge_state
        edge_status.values = [
            KeyValue(key="enabled", value=str(edge_enabled).lower()),
            KeyValue(key="state", value=edge_state),
            KeyValue(key="error", value=edge_error),
            KeyValue(
                key="latency_ms",
                value=f"{self._latest_edge_latency_ms:.3f}",
            ),
            KeyValue(
                key="refined_corrections",
                value=str(edge_refined_corrections),
            ),
            KeyValue(
                key="stale_results",
                value=str(edge_stale_results),
            ),
            KeyValue(
                key="retained_contexts",
                value=str(retained_edge_contexts),
            ),
            *[
                KeyValue(key=key, value=values[key])
                for key in (
                    "track_count",
                    "prompt_count",
                    "hand_candidate_count",
                    "geometry_fallback_track_count",
                    "hand_semantic_status",
                    "hand_rgb_detection_count",
                    "hand_semantic_reject_reasons",
                    "safety_output_state",
                    "background_state",
                    "background_alignment_valid",
                )
                if key in values
            ],
            *[
                KeyValue(key=key, value=value)
                for key, value in sorted(mask_diagnostics.items())
            ],
        ]
        diagnostic.status = [status, edge_status]
        self._diagnostics_publisher.publish(diagnostic)
        self._fps_publisher.publish(Float32(data=float(fps)))
        self._latency_publisher.publish(Float32(data=float(latency)))

    def destroy_node(self) -> bool:
        self._shutdown.set()
        try:
            self._work_queue.put_nowait(None)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
                self._work_queue.task_done()
                self._work_queue.put_nowait(None)
            except queue.Empty:
                pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if (
            self._edge_loader is not None
            and self._edge_loader.is_alive()
        ):
            self._edge_loader.join(timeout=2.0)
        if self._edge is not None:
            self._edge.close(timeout=2.0)
        if self._hand_detector is not None:
            self._hand_detector.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: EdgeTAMPointCloudTrackerNode | None = None
    try:
        node = EdgeTAMPointCloudTrackerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = ["EdgeTAMPointCloudTrackerNode", "main"]
