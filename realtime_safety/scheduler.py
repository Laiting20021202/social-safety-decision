from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from realtime_safety.config import AppConfig
from realtime_safety.gui.gui_state import GuiState
from realtime_safety.gui.video_overlay import draw_video_overlay
from realtime_safety.performance import PerformanceMonitor
from realtime_safety.pipeline.camera_motion import CameraMotionEstimator
from realtime_safety.pipeline.danger_zone import DangerZonePredictor
from realtime_safety.pipeline.frame_queue import LatestQueue
from realtime_safety.pipeline.ground_plane import GroundPlaneEstimator
from realtime_safety.pipeline.local_planner import LocalSafetyPlanner, PlannerResult
from realtime_safety.pipeline.monocular_depth import MonocularDepthBackend
from realtime_safety.pipeline.obstacle_3d import ObstacleExtractor3D
from realtime_safety.pipeline.pointcloud import voxel_downsample
from realtime_safety.pipeline.robot_self_filter import RobotSelfFilter
from realtime_safety.pipeline.safety_decision import SafetyDecisionEngine
from realtime_safety.pipeline.segmentation import SegmentationBackend, create_segmentation_backend
from realtime_safety.pipeline.st4rtrack_adapter import St4RTrackAdapter
from realtime_safety.pipeline.tracker_2d import StableTracker2D
from realtime_safety.pipeline.tracker_3d import Tracker3D
from realtime_safety.pipeline.traversable_region import TraversableRegion, compute_traversable_region
from realtime_safety.pipeline.video_source import AUTO_CAMERA_ALIASES, CameraDetectionError, PlaybackState, VideoSource
from realtime_safety.pipeline.video_depth import VideoDepthBackend
from realtime_safety.types import (
    BBox3D,
    Detection2D,
    FramePacket,
    ObstacleObservation3D,
    PipelineSnapshot,
    PointCloudFrame,
    RecommendedAction,
    RobotArmState,
    SafetyLevel,
    Track3DState,
)
from realtime_safety.utils.gpu import gpu_info, release_gpu_memory

LOGGER = logging.getLogger(__name__)


_OBSTACLE_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (255, 64, 64),
    "bicycle": (255, 160, 32),
    "motorcycle": (255, 192, 32),
    "vehicle": (255, 224, 32),
    "chair": (192, 96, 255),
    "bag": (255, 96, 192),
    "suitcase": (96, 192, 255),
    "bench": (96, 255, 160),
    "dog": (64, 224, 255),
    "cat": (64, 224, 255),
}


def _observation_pointcloud(
    observations: list[ObstacleObservation3D],
    source_cloud: PointCloudFrame,
    max_points: int,
) -> PointCloudFrame:
    """Aggregate YOLO-mask 3D clusters into one color-coded PointCloudFrame."""

    point_groups: list[np.ndarray] = []
    color_groups: list[np.ndarray] = []
    confidence_groups: list[np.ndarray] = []
    for observation in observations:
        if observation.points is None or len(observation.points) == 0:
            continue
        points = np.asarray(observation.points, dtype=np.float32).reshape(-1, 3)
        color = _OBSTACLE_COLORS.get(observation.class_name, (64, 255, 255))
        point_groups.append(points)
        color_groups.append(np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)))
        confidence_groups.append(
            np.full(len(points), observation.confidence, dtype=np.float32)
        )

    if point_groups:
        points = np.concatenate(point_groups, axis=0)
        colors = np.concatenate(color_groups, axis=0)
        confidence = np.concatenate(confidence_groups, axis=0)
        if len(points) > max_points:
            selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points, colors, confidence = (
                points[selected],
                colors[selected],
                confidence[selected],
            )
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
        confidence = np.empty((0,), dtype=np.float32)

    return PointCloudFrame(
        points=points,
        colors=colors,
        confidence=confidence,
        pointmap=source_cloud.pointmap,
        frame_index=source_cloud.frame_index,
        timestamp=source_cloud.timestamp,
        anchor_frame_index=source_cloud.anchor_frame_index,
        inference_ms=source_cloud.inference_ms,
        valid=len(points) > 0,
        source="yolo_obstacles",
        metric_scale=source_cloud.metric_scale,
        reference_depth_m=source_cloud.reference_depth_m,
        reference_observed_depth=source_cloud.reference_observed_depth,
    )


def _observations_with_short_hold(
    observations: list[ObstacleObservation3D],
    tracks: list[Track3DState],
    cache: dict[int, ObstacleObservation3D],
    max_missing: int,
    timestamp: float,
    voxel_size: float = 0.0,
    max_center_step_m: float = 0.18,
) -> list[ObstacleObservation3D]:
    """Stabilize measured clusters and keep them across brief detector misses."""

    result: list[ObstacleObservation3D] = []
    observed_ids = {observation.track_id for observation in observations}
    for observation in observations:
        previous = cache.get(observation.track_id)
        stabilized = _fuse_obstacle_observation(
            previous,
            observation,
            voxel_size=voxel_size,
            max_center_step_m=max_center_step_m,
        )
        cache[observation.track_id] = stabilized
        result.append(stabilized)

    active_ids = {track.track_id for track in tracks}
    for track in tracks:
        if (
            track.track_id in observed_ids
            or track.missing_count <= 0
            or track.missing_count > max_missing
        ):
            continue
        previous = cache.get(track.track_id)
        if previous is None or previous.points is None or not len(previous.points):
            continue
        displacement = np.asarray(track.position_xyz - previous.position_xyz, dtype=np.float32)
        distance = float(np.linalg.norm(displacement))
        # Short holds prevent flicker, but a noisy monocular velocity must not
        # launch a stale obstacle cluster across the scene.
        if distance > 0.35:
            displacement *= 0.35 / distance
        points = np.asarray(previous.points, dtype=np.float32) + displacement
        result.append(
            ObstacleObservation3D(
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=max(0.05, previous.confidence * 0.85**track.missing_count),
                position_xyz=previous.position_xyz + displacement,
                bbox3d=BBox3D(
                    minimum=previous.bbox3d.minimum + displacement,
                    maximum=previous.bbox3d.maximum + displacement,
                ),
                radius=previous.radius,
                point_count=len(points),
                timestamp=timestamp,
                points=points,
            )
        )
        cache[track.track_id] = result[-1]

    for track_id in tuple(cache):
        if track_id not in active_ids:
            del cache[track_id]
    return result


def _fuse_obstacle_observation(
    previous: ObstacleObservation3D | None,
    current: ObstacleObservation3D,
    voxel_size: float,
    max_center_step_m: float = 0.18,
    max_points: int = 3000,
) -> ObstacleObservation3D:
    """Fill transient mask/depth holes using the previous aligned track cloud."""

    if (
        previous is None
        or previous.points is None
        or not len(previous.points)
        or current.points is None
        or not len(current.points)
        or previous.class_name != current.class_name
    ):
        return current
    displacement = np.asarray(
        current.position_xyz - previous.position_xyz,
        dtype=np.float32,
    )
    distance = float(np.linalg.norm(displacement))
    maximum_step = max(
        float(max_center_step_m),
        0.75 * min(float(current.radius), float(previous.radius)),
    )
    if distance > maximum_step:
        # Monocular depth occasionally moves a whole mask layer between two
        # frames. Rate-limit that geometry jump instead of flashing the cloud
        # at two unrelated depths.
        limited = displacement * (maximum_step / distance)
        corrected_center = previous.position_xyz + limited
        correction = corrected_center - current.position_xyz
        corrected_points = (
            np.asarray(current.points, dtype=np.float32).reshape(-1, 3)
            + correction
        )
        current = ObstacleObservation3D(
            track_id=current.track_id,
            class_name=current.class_name,
            confidence=current.confidence,
            position_xyz=corrected_center.astype(np.float32),
            bbox3d=BBox3D(
                minimum=current.bbox3d.minimum + correction,
                maximum=current.bbox3d.maximum + correction,
            ),
            radius=current.radius,
            point_count=len(corrected_points),
            timestamp=current.timestamp,
            points=corrected_points,
        )
        displacement = limited
    # A large jump is more likely an ID switch or a bad monocular depth frame;
    # never smear the old obstacle across that jump.
    maximum_fusion_shift = max(0.20, 1.5 * current.radius)
    if float(np.linalg.norm(displacement)) > maximum_fusion_shift:
        return current

    current_points = np.asarray(current.points, dtype=np.float32).reshape(-1, 3)
    previous_points = (
        np.asarray(previous.points, dtype=np.float32).reshape(-1, 3) + displacement
    )
    margin = max(0.04, min(0.12, 0.2 * current.radius))
    lower = current.bbox3d.minimum - margin
    upper = current.bbox3d.maximum + margin
    previous_points = previous_points[
        np.all((previous_points >= lower) & (previous_points <= upper), axis=1)
    ]
    if not len(previous_points):
        return current

    # Current points are first, so deterministic voxel selection always favors
    # new geometry; old geometry only fills holes inside the current volume.
    combined = np.concatenate((current_points, previous_points), axis=0)
    confidence = np.concatenate(
        (
            np.full(len(current_points), current.confidence, dtype=np.float32),
            np.full(
                len(previous_points),
                previous.confidence * 0.8,
                dtype=np.float32,
            ),
        )
    )
    points, _, _ = voxel_downsample(
        combined,
        np.zeros((len(combined), 3), dtype=np.uint8),
        confidence,
        max(float(voxel_size), 0.0),
        max_points=max_points,
    )
    minimum, maximum = np.percentile(points, (5.0, 95.0), axis=0).astype(np.float32)
    return ObstacleObservation3D(
        track_id=current.track_id,
        class_name=current.class_name,
        confidence=current.confidence,
        position_xyz=current.position_xyz.copy(),
        bbox3d=BBox3D(minimum=minimum, maximum=maximum),
        radius=current.radius,
        point_count=len(points),
        timestamp=current.timestamp,
        points=points,
    )


@dataclass(slots=True)
class PerceptionResult:
    frame: FramePacket
    detections: list[Detection2D]
    camera_motion_confidence: float


class AdaptiveRealtimeController:
    """Conservative quality ladder that never lowers the safety tick rate."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.gpu = gpu_info(config.device)
        self.profile = config.name.removeprefix("realtime_").upper()
        self._base_points = config.reconstruction.max_points
        self._base_reconstruction_hz = config.reconstruction.frequency_hz
        self._pressure_count = 0
        self._stable_count = 0
        self._last_adjustment = 0.0

    def startup_calibration(self, segmentation_ms: float | None, reconstruction_ms: float | None) -> None:
        if self.gpu.available and self.gpu.total_mb < 6000:
            self._degrade()
        if segmentation_ms and segmentation_ms > 100:
            self.config.segmentation.input_size = max(256, min(self.config.segmentation.input_size, 320))
        if reconstruction_ms and reconstruction_ms > 450:
            self.config.reconstruction.frequency_hz = min(self.config.reconstruction.frequency_hz, 1.5)

    def observe(self, performance) -> str:
        now = time.perf_counter()
        pressure = (
            performance.p95_latency_ms > 300.0
            or (performance.queue_capacity > 0 and performance.queue_size >= performance.queue_capacity)
            or (performance.display_fps > 0 and performance.display_fps < 10.0)
            or (self.gpu.total_mb > 0 and performance.vram_used_mb > self.gpu.total_mb * 0.9)
        )
        if pressure:
            self._pressure_count += 1
            self._stable_count = 0
        else:
            self._stable_count += 1
            self._pressure_count = 0
        if self._pressure_count >= 3 and now - self._last_adjustment > 2.0:
            self._degrade()
            self._last_adjustment = now
            self._pressure_count = 0
        elif self._stable_count >= 50 and now - self._last_adjustment > 10.0:
            self._recover()
            self._last_adjustment = now
            self._stable_count = 0
        return self.profile

    def _degrade(self) -> None:
        if self.config.reconstruction.max_points > 10_000:
            self.config.reconstruction.max_points = max(10_000, int(self.config.reconstruction.max_points * 0.7))
        elif self.config.reconstruction.input_size > 224:
            self.config.reconstruction.input_size = 224
        elif self.config.segmentation.input_size > 320:
            self.config.segmentation.input_size = 320
        elif self.config.reconstruction.frequency_hz > 1.0:
            self.config.reconstruction.frequency_hz = max(1.0, self.config.reconstruction.frequency_hz * 0.75)
        elif self.config.segmentation.frequency_hz > 5.0:
            self.config.segmentation.frequency_hz = max(5.0, self.config.segmentation.frequency_hz * 0.8)
        self.profile = "DEGRADED"

    def _recover(self) -> None:
        self.config.reconstruction.max_points = min(
            self._base_points, int(max(self.config.reconstruction.max_points + 1, self.config.reconstruction.max_points * 1.2))
        )
        self.config.reconstruction.frequency_hz = min(
            self._base_reconstruction_hz, self.config.reconstruction.frequency_hz * 1.1
        )
        if self.config.reconstruction.max_points >= self._base_points:
            self.profile = self.config.name.removeprefix("realtime_").upper()


class RealtimePipeline:
    def __init__(
        self,
        config: AppConfig,
        segmentation_backend: SegmentationBackend | None = None,
        depth_backend: MonocularDepthBackend | None = None,
        dashboard: Any | None = None,
        scene: Any | None = None,
        session_logger: Any | None = None,
        video_recorder: Any | None = None,
        pointcloud_publisher: Any | None = None,
        yolo_obstacle_pointcloud_publisher: Any | None = None,
        arm_obstacle_relationship_publisher: Any | None = None,
        camera_preview_publisher: Any | None = None,
    ) -> None:
        self.config = config
        self.segmentation = segmentation_backend
        if self.segmentation is None and (
            config.mode == "safety"
            or config.people_overlay
            or yolo_obstacle_pointcloud_publisher is not None
            or arm_obstacle_relationship_publisher is not None
        ):
            self.segmentation = create_segmentation_backend(config.segmentation, config.device)
        self.depth = depth_backend or (
            VideoDepthBackend(config.reconstruction, config.device)
            if config.reconstruction.depth_mode == "video_depth"
            else MonocularDepthBackend(config.reconstruction, config.device)
        )
        self.st4r = St4RTrackAdapter(config.reconstruction, config.device)
        self.dashboard = dashboard
        self.scene = scene
        self.session_logger = session_logger
        self.video_recorder = video_recorder
        self.pointcloud_publisher = pointcloud_publisher
        self.yolo_obstacle_pointcloud_publisher = yolo_obstacle_pointcloud_publisher
        self.arm_obstacle_relationship_publisher = (
            arm_obstacle_relationship_publisher
        )
        self.camera_preview_publisher = camera_preview_publisher
        self.gui_state = GuiState()
        self.performance = PerformanceMonitor(config.device)
        self.adaptive = AdaptiveRealtimeController(config)
        self.capture_queue: LatestQueue[FramePacket] = LatestQueue(config.video.queue_size)
        self.reconstruction_queue: LatestQueue[FramePacket] = LatestQueue(config.video.queue_size)
        self._state_lock = threading.RLock()
        self._source_lock = threading.RLock()
        self._perceptions: deque[PerceptionResult] = deque(maxlen=4)
        self._latest_capture: FramePacket | None = None
        self._cloud: PointCloudFrame | None = None
        self._cloud_source_frame: FramePacket | None = None
        self._cloud_version = 0
        self._people_tracks: list[Track3DState] = []
        self._people_detections: list[Detection2D] = []
        self._robot_arm: RobotArmState | None = None
        self._people_detection_frame: FramePacket | None = None
        self._people_version = 0
        self._people_cloud_version = -1
        self._people_frame_index = -1
        self._safety = None
        self._safety_cloud_version = -1
        self._traversable = TraversableRegion(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0.0)
        self._planner_result = PlannerResult([], None, RecommendedAction.WAIT)
        self._source: VideoSource | None = None
        self._stop_event = threading.Event()
        self._new_frame_event = threading.Event()
        self._source_done = threading.Event()
        self._capture_complete = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._max_frames: int | None = None
        self._captured_frames = 0
        self._errors: dict[str, str] = {}
        self._gpu_lock = threading.Lock()
        self._models_ready = {"segmentation": False, "depth": False, "st4rtrack": False}
        self._segmentation_ms: float | None = None
        self._reconstruction_ms: float | None = None
        self._final_frame_index = -1
        self._calibrated = False
        self._reset_counter = 0

    @property
    def errors(self) -> dict[str, str]:
        with self._state_lock:
            return dict(self._errors)

    @property
    def source_done(self) -> bool:
        return self._source_done.is_set()

    def start_workers(self) -> None:
        if self._running:
            return
        if self.pointcloud_publisher is not None:
            self.pointcloud_publisher.start()
        if self.yolo_obstacle_pointcloud_publisher is not None:
            self.yolo_obstacle_pointcloud_publisher.start()
        if self.arm_obstacle_relationship_publisher is not None:
            self.arm_obstacle_relationship_publisher.start()
        if self.camera_preview_publisher is not None:
            self.camera_preview_publisher.start()
        self._stop_event.clear()
        self._new_frame_event.clear()
        if self.config.mode == "reconstruction":
            targets_list = [
                ("capture-worker", self._capture_worker),
                ("3d-reconstruction-worker", self._reconstruction_worker),
            ]
            if (
                self.config.people_overlay
                or self.yolo_obstacle_pointcloud_publisher is not None
                or self.arm_obstacle_relationship_publisher is not None
            ):
                targets_list.append(("people-3d-worker", self._people_worker))
            if self.dashboard is not None or self.video_recorder is not None:
                # Point-cloud serialization can block the scene renderer.  Keep
                # the live RGB path independent so camera playback remains
                # smooth while 3D and YOLO updates are being sent.
                targets_list.append(("gui-video-renderer", self._video_renderer_worker))
            targets_list.append(("gui-renderer", self._renderer_worker))
            targets = tuple(targets_list)
        else:
            targets = (
                ("capture-worker", self._capture_worker),
                ("fast-perception-worker", self._fast_worker),
                ("3d-reconstruction-worker", self._reconstruction_worker),
                ("safety-worker", self._safety_worker),
                ("gui-renderer", self._renderer_worker),
            )
        self._threads = [threading.Thread(name=name, target=target, daemon=True) for name, target in targets]
        for thread in self._threads:
            thread.start()
        self._running = True

    def start_source(self, source: str | int, max_frames: int | None = None) -> None:
        video = VideoSource(source, loop=self.config.video.loop, playback_speed=self.config.video.playback_speed)
        video.open()
        with self._source_lock:
            if self._source is not None:
                self._source.close()
            self._source = video
            self._max_frames = max_frames
            self._captured_frames = 0
            self._source_done.clear()
            self._capture_complete.clear()
            self._final_frame_index = -1
        self._reset_runtime_state()
        if video.camera_info is not None and self.dashboard is not None:
            camera = video.camera_info
            self.dashboard.update_camera_status(
                f"Webcam: **connected** — {camera.description}  \n"
                f"RGB frames are feeding the **{self.config.reconstruction.depth_mode}** depth pipeline.",
                camera.index,
                connected=True,
            )
        elif video.is_remote_stream and self.dashboard is not None:
            connection_state = "connected" if video.is_connected else "reconnecting"
            transport = "ROS 2 camera" if video.is_ros2_stream else "Network stream"
            self.dashboard.update_camera_status(
                f"{transport}: **{connection_state}**  \n"
                f"RGB frames are feeding the **{self.config.reconstruction.depth_mode}** depth pipeline.",
                video.source,
                connected=video.is_connected,
            )
        source_kind = (
            "webcam"
            if video.camera_info is not None
            else "ROS 2 camera"
            if video.is_ros2_stream
            else "network stream"
            if video.is_network_stream
            else "video"
        )
        LOGGER.info(
            "Started %s source %s",
            source_kind,
            video.camera_info.description if video.camera_info is not None else video.source,
        )

    def _reset_runtime_state(self) -> None:
        self.capture_queue.clear()
        self.reconstruction_queue.clear()
        with self._state_lock:
            self._reset_counter += 1
            self._latest_capture = None
            self._perceptions.clear()
            self._cloud = None
            self._cloud_source_frame = None
            self._cloud_version = 0
            self._people_tracks = []
            self._people_detections = []
            self._robot_arm = None
            self._people_detection_frame = None
            self._people_version = 0
            self._people_cloud_version = -1
            self._people_frame_index = -1
            self._safety = None
            self._safety_cloud_version = -1
            self._traversable = TraversableRegion(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0.0)
            self._planner_result = PlannerResult([], None, RecommendedAction.WAIT)
        if self.scene is not None and hasattr(self.scene, "reset"):
            self.scene.reset()

    def handle_command(self, command: str, value: object | None = None) -> None:
        if command == "camera_fov" and isinstance(value, tuple) and len(value) == 4:
            horizontal_fov, vertical_fov, image_width, image_height = map(float, value)
            if not (1.0 < horizontal_fov < 179.0 and 1.0 < vertical_fov < 179.0):
                LOGGER.warning("Ignoring invalid camera FOV: %s", value)
                return
            reconstruction = self.config.reconstruction
            reconstruction.focal_length_x = image_width / (
                2.0 * math.tan(math.radians(horizontal_fov) / 2.0)
            )
            reconstruction.focal_length_y = image_height / (
                2.0 * math.tan(math.radians(vertical_fov) / 2.0)
            )
            reconstruction.principal_point_x = (image_width - 1.0) / 2.0
            reconstruction.principal_point_y = (image_height - 1.0) / 2.0
            LOGGER.info(
                "Updated camera projection: %.1fx%.1f deg, fx=%.2f, fy=%.2f at %.0fx%.0f",
                horizontal_fov,
                vertical_fov,
                reconstruction.focal_length_x,
                reconstruction.focal_length_y,
                image_width,
                image_height,
            )
            self._reset_runtime_state()
            return
        if command in {"start", "detect_camera"}:
            raw_source = "auto" if command == "detect_camera" or value in (None, "") else str(value).strip()
            source: str | int = int(raw_source) if raw_source.isdigit() else raw_source
            camera_request = (
                command == "detect_camera"
                or raw_source.lower() in AUTO_CAMERA_ALIASES
                or raw_source.isdigit()
                or raw_source.startswith("/dev/video")
            )
            if camera_request:
                with self._source_lock:
                    if self._source is not None and self._source.camera_info is not None:
                        self._source.close()
            try:
                self.start_source(source)
            except (CameraDetectionError, FileNotFoundError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Could not start input source %r: %s", source, exc)
                if self.dashboard is not None:
                    self.dashboard.update_camera_status(f"Webcam/input error: **{exc}**")
            return
        reset_runtime = False
        with self._source_lock:
            source = self._source
            if command == "pause_resume" and source:
                source.resume() if source.state == PlaybackState.PAUSED else source.pause()
            elif command == "stop" and source:
                source.stop()
                self._capture_complete.set()
                self._source_done.set()
            elif command == "restart" and source:
                source.restart()
                self._source_done.clear()
                self._capture_complete.clear()
                reset_runtime = True
            elif command == "loop" and source:
                source.loop = bool(value)
            elif command == "speed" and source:
                source.set_playback_speed(float(value))
            elif command == "seek" and source:
                source.seek(float(value))
                reset_runtime = True
        if reset_runtime:
            self._reset_runtime_state()
        if command == "visibility" and self.scene and isinstance(value, tuple):
            self.scene.set_visibility(str(value[0]), bool(value[1]))

    def wait_until_source_done(self, timeout: float | None = None) -> bool:
        return self._source_done.wait(timeout)

    def close(self) -> None:
        self._stop_event.set()
        self._new_frame_event.set()
        with self._source_lock:
            if self._source:
                self._source.close()
        for thread in self._threads:
            thread.join(timeout=10.0)
        self._threads.clear()
        self._running = False
        if self.pointcloud_publisher is not None:
            self.pointcloud_publisher.close()
        if self.yolo_obstacle_pointcloud_publisher is not None:
            self.yolo_obstacle_pointcloud_publisher.close()
        if self.arm_obstacle_relationship_publisher is not None:
            self.arm_obstacle_relationship_publisher.close()
        if self.camera_preview_publisher is not None:
            self.camera_preview_publisher.close()
        release_gpu_memory()
        if self.session_logger is not None:
            self.session_logger.close()
        if self.video_recorder is not None:
            self.video_recorder.close()

    def _capture_worker(self) -> None:
        next_deadline = time.perf_counter()
        last_source_identity = None
        last_frame_index = -1
        remote_connected: bool | None = None
        while not self._stop_event.is_set():
            with self._source_lock:
                source = self._source
            if source is None or source.state in (PlaybackState.STOPPED, PlaybackState.PAUSED, PlaybackState.ENDED):
                if source is not None and source.state == PlaybackState.ENDED:
                    self._capture_complete.set()
                self._stop_event.wait(0.01)
                continue
            if id(source) != last_source_identity:
                next_deadline = time.perf_counter()
                last_source_identity = id(source)
                last_frame_index = -1
                remote_connected = None
            wait = next_deadline - time.perf_counter()
            if wait > 0 and self._stop_event.wait(wait):
                break
            frame = source.read()
            if frame is None:
                if source.is_remote_stream and not source.is_connected and remote_connected is not False:
                    remote_connected = False
                    if self.dashboard is not None:
                        transport = "ROS 2 camera" if source.is_ros2_stream else "Network stream"
                        self.dashboard.update_camera_status(
                            f"{transport}: **reconnecting** — no fresh camera frames.  \n"
                            "The GUI remains online and will resume automatically.",
                            source.source,
                            connected=False,
                        )
                # A disconnected remote source may be waiting for DDS data or
                # an HTTP reconnect. Avoid burning a CPU core in either case.
                self._stop_event.wait(0.05)
                continue
            if source.is_remote_stream and remote_connected is not True:
                remote_connected = True
                if self.dashboard is not None:
                    transport = "ROS 2 camera" if source.is_ros2_stream else "Network stream"
                    self.dashboard.update_camera_status(
                        f"{transport}: **connected** — {frame.original_width}x{frame.original_height}.  \n"
                        f"RGB frames are feeding the **{self.config.reconstruction.depth_mode}** depth pipeline.",
                        source.source,
                        connected=True,
                    )
            if frame.frame_index <= last_frame_index:
                self._reset_runtime_state()
            last_frame_index = frame.frame_index
            if self.config.mode == "safety":
                self.capture_queue.put_latest(frame)
            self.reconstruction_queue.put_latest(frame)
            self.performance.tick("input")
            with self._state_lock:
                self._latest_capture = frame
            if self.camera_preview_publisher is not None:
                self.camera_preview_publisher.publish(frame)
            self._new_frame_event.set()
            self._captured_frames += 1
            self._final_frame_index = frame.frame_index
            if self._max_frames is not None and self._captured_frames >= self._max_frames:
                source.stop()
                self._capture_complete.set()
            fps = frame.original_fps
            interval = 1.0 / (fps * source.playback_speed) if fps > 0 and not source.is_live else 0.0
            next_deadline = max(next_deadline + interval, time.perf_counter() - interval)

    def _fast_worker(self) -> None:
        if self.segmentation is None:
            return
        tracker = StableTracker2D(self.config.tracking)
        robot_self_filter = RobotSelfFilter(self.config.segmentation)
        camera_motion = CameraMotionEstimator()
        try:
            self.segmentation.load()
            with self._gpu_lock:
                self.segmentation.warmup()
            self._models_ready["segmentation"] = True
        except Exception as exc:
            LOGGER.exception("Segmentation initialization failed")
            self._errors["segmentation"] = str(exc)
        last_segmentation = 0.0
        reset_counter = self._reset_counter
        while not self._stop_event.is_set():
            try:
                frame = self.capture_queue.get_latest(timeout=0.05)
            except queue.Empty:
                continue
            if reset_counter != self._reset_counter:
                tracker.reset()
                robot_self_filter.reset()
                camera_motion.reset()
                reset_counter = self._reset_counter
                last_segmentation = 0.0
            now = time.perf_counter()
            run_segmentation = self._models_ready["segmentation"] and now - last_segmentation >= 1.0 / max(self.config.segmentation.frequency_hz, 0.1)
            if run_segmentation:
                start = time.perf_counter()
                try:
                    with self._gpu_lock:
                        inference = self.segmentation.infer(frame)
                    inference = robot_self_filter.filter_people(inference, frame)
                    detections = tracker.update(inference, frame.source_timestamp)
                    self.performance.tick("segmentation")
                    self._segmentation_ms = (time.perf_counter() - start) * 1000.0
                except Exception as exc:
                    self._errors["segmentation_runtime"] = str(exc)
                    detections = tracker.predict(frame.source_timestamp)
                last_segmentation = now
            else:
                detections = tracker.predict(frame.source_timestamp)
            exclusion = np.zeros(frame.bgr.shape[:2], dtype=bool)
            for detection in detections:
                if detection.mask is not None and detection.mask.shape == exclusion.shape:
                    exclusion |= detection.mask
            motion = camera_motion.update(frame.bgr, exclusion)
            with self._state_lock:
                self._perceptions.append(PerceptionResult(frame, detections, motion.confidence))
        self.segmentation.close()

    def _reconstruction_worker(self) -> None:
        depth_mode = self.config.reconstruction.depth_mode
        needs_depth_fallback = depth_mode != "st4rtrack" or self.config.mode == "safety"
        if needs_depth_fallback:
            try:
                self.depth.load()
                with self._gpu_lock:
                    self.depth.warmup()
                self._models_ready["depth"] = True
            except Exception as exc:
                if depth_mode == "video_depth":
                    LOGGER.warning("Video depth unavailable; using per-frame Depth Anything V2 fallback: %s", exc)
                    self._errors["video_depth"] = str(exc)
                    self.depth = MonocularDepthBackend(self.config.reconstruction, self.config.device)
                    try:
                        self.depth.load()
                        with self._gpu_lock:
                            self.depth.warmup()
                        self._models_ready["depth"] = True
                    except Exception as fallback_exc:
                        LOGGER.exception("Depth fallback initialization failed")
                        self._errors["depth"] = str(fallback_exc)
                else:
                    LOGGER.exception("Depth initialization failed")
                    self._errors["depth"] = str(exc)
        if depth_mode in {"st4rtrack", "hybrid"}:
            try:
                self.st4r.load()
                with self._gpu_lock:
                    self.st4r.warmup()
                self._models_ready["st4rtrack"] = True
            except Exception as exc:
                LOGGER.warning("St4RTrack unavailable; using fast-depth fallback: %s", exc)
                self._errors["st4rtrack"] = str(exc)
        anchor: FramePacket | None = None
        runtime_stream = _make_cuda_stream(self.config.device) if self.config.mode == "reconstruction" else None
        last_depth = 0.0
        last_st4r = 0.0
        pending_frame: FramePacket | None = None
        reset_counter = self._reset_counter
        while not self._stop_event.is_set():
            if self.config.people_overlay and not self._models_ready["segmentation"] and "segmentation" not in self._errors:
                self._stop_event.wait(0.01)
                continue
            if pending_frame is None:
                try:
                    frame = self.reconstruction_queue.get_latest(timeout=0.05)
                except queue.Empty:
                    continue
            else:
                frame = pending_frame
                try:
                    frame = self.reconstruction_queue.get_latest(timeout=0.0)
                except queue.Empty:
                    pass
            if reset_counter != self._reset_counter:
                anchor = None
                pending_frame = None
                self.st4r.reset()
                if hasattr(self.depth, "reset"):
                    self.depth.reset()
                reset_counter = self._reset_counter
            now = time.perf_counter()
            st4r_due = now - last_st4r >= 1.0 / max(self.config.reconstruction.frequency_hz, 0.1)
            depth_rate = (
                self.config.reconstruction.fast_depth_frequency_hz
                if depth_mode in {"hybrid", "fast_depth"}
                else self.config.reconstruction.frequency_hz
            )
            depth_due = now - last_depth >= 1.0 / max(depth_rate, 0.1)
            use_st4r = self._models_ready["st4rtrack"] and depth_mode in {"st4rtrack", "hybrid"} and st4r_due
            use_depth = self._models_ready["depth"] and depth_due and (depth_mode != "st4rtrack" or not self._models_ready["st4rtrack"])
            if not use_st4r and not use_depth:
                pending_frame = frame
                self._stop_event.wait(0.005)
                continue
            pending_frame = None
            start = time.perf_counter()
            try:
                if use_st4r:
                    if anchor is None:
                        anchor = frame
                        self.st4r.set_anchor(frame)
                    if runtime_stream is None:
                        with self._gpu_lock:
                            cloud = self.st4r.infer(anchor, frame)
                    else:
                        import torch

                        with torch.cuda.stream(runtime_stream):
                            cloud = self.st4r.infer(anchor, frame)
                    # Rate limits are start-to-start. Measuring from inference
                    # completion needlessly added a full interval after every
                    # model call and cut the achieved update rate in half.
                    last_st4r = start
                    anchor_interval = self.config.reconstruction.anchor_interval
                    if anchor_interval > 0 and frame.frame_index - anchor.frame_index > anchor_interval:
                        anchor = frame
                        self.st4r.set_anchor(frame)
                elif use_depth:
                    if runtime_stream is None:
                        with self._gpu_lock:
                            cloud = self.depth.infer(frame)
                    else:
                        import torch

                        with torch.cuda.stream(runtime_stream):
                            cloud = self.depth.infer(frame)
                    last_depth = start
                else:
                    continue
                if self.config.manual_scale is not None:
                    cloud.points *= self.config.manual_scale
                    cloud.pointmap *= self.config.manual_scale
                    if cloud.tracking_points is not None:
                        cloud.tracking_points *= self.config.manual_scale
                cloud.points = cloud.points[: self.config.reconstruction.max_points]
                cloud.colors = cloud.colors[: len(cloud.points)]
                cloud.confidence = cloud.confidence[: len(cloud.points)]
                if self.pointcloud_publisher is not None:
                    try:
                        self.pointcloud_publisher.publish(cloud)
                    except Exception as publish_exc:
                        if "pointcloud_topic" not in self._errors:
                            LOGGER.exception("Point-cloud topic publication failed")
                            self._errors["pointcloud_topic"] = str(publish_exc)
                self.performance.tick("reconstruction")
                self._reconstruction_ms = (time.perf_counter() - start) * 1000.0
                with self._state_lock:
                    self._cloud = cloud
                    self._cloud_source_frame = frame
                    self._cloud_version += 1
            except Exception as exc:
                LOGGER.exception("Reconstruction failed")
                self._errors["reconstruction_runtime"] = str(exc)
        self.st4r.close()
        self.depth.close()

    def _people_worker(self) -> None:
        if self.segmentation is None:
            return
        maximum_depth = (
            self.config.reconstruction.max_metric_depth_m
            or self.config.reconstruction.max_relative_depth
        )
        extractor = ObstacleExtractor3D(
            self.config.reconstruction.confidence_threshold,
            maximum_depth,
            self.config.reconstruction.voxel_size,
            minimum_points=20,
        )
        tracker_2d = StableTracker2D(self.config.tracking)
        tracker_3d = Tracker3D(self.config.tracking)
        robot_self_filter = RobotSelfFilter(self.config.segmentation)
        observation_cache: dict[int, ObstacleObservation3D] = {}
        native_obstacle_tracker = callable(
            getattr(self.segmentation, "track_obstacles", None)
        )
        try:
            self.segmentation.load()
            with self._gpu_lock:
                self.segmentation.warmup()
            self._models_ready["segmentation"] = True
        except Exception as exc:
            LOGGER.exception("YOLO person initialization failed")
            self._errors["segmentation"] = str(exc)

        runtime_stream = _make_cuda_stream(self.config.device)

        handled_cloud_version = -1
        reset_counter = self._reset_counter
        last_yolo_diagnostics_log = 0.0
        while not self._stop_event.is_set():
            with self._state_lock:
                cloud = self._cloud
                source_frame = self._cloud_source_frame
                cloud_version = self._cloud_version
            if cloud is None or source_frame is None or cloud_version == handled_cloud_version:
                self._stop_event.wait(0.01)
                continue
            if reset_counter != self._reset_counter:
                tracker_2d.reset()
                tracker_3d.reset()
                robot_self_filter.reset()
                observation_cache.clear()
                reset_tracking = getattr(self.segmentation, "reset_tracking", None)
                if callable(reset_tracking):
                    reset_tracking()
                handled_cloud_version = -1
                reset_counter = self._reset_counter
            try:
                people_2d: list[Detection2D] = []
                robot_arm: RobotArmState | None = None
                if self._models_ready["segmentation"]:
                    start = time.perf_counter()

                    def run_inference() -> list[Detection2D]:
                        if native_obstacle_tracker:
                            return self.segmentation.track_obstacles(source_frame)
                        return self.segmentation.infer(source_frame)

                    if runtime_stream is None:
                        with self._gpu_lock:
                            raw_inference = run_inference()
                    else:
                        import torch

                        with torch.cuda.stream(runtime_stream):
                            raw_inference = run_inference()
                    inference = robot_self_filter.filter_obstacles(
                        raw_inference,
                        source_frame,
                    )
                    robot_arm = robot_self_filter.estimate_arm_state(
                        source_frame,
                        cloud,
                    )
                    # ByteTrack supplies low-confidence recovery masks, while
                    # this local timestamp-aware association keeps IDs stable
                    # if the backend resets or changes an external ID.
                    measured_people = tracker_2d.update(
                        inference,
                        source_frame.source_timestamp,
                    )
                    held_people = tracker_2d.predict_missing(
                        source_frame.source_timestamp,
                        self.config.tracking.visual_hold_updates,
                    )
                    # One-frame low-confidence boxes are useful to ByteTrack
                    # internally but visually look like false-positive flashes.
                    # Publish/render only tracks that passed the same consecutive
                    # hit gate used by the 3D obstacle extractor.
                    visible_measured = [
                        detection
                        for detection in measured_people
                        if detection.track_hits
                        >= self.config.tracking.confirmation_hits
                    ]
                    visible_held = [
                        detection
                        for detection in held_people
                        if detection.track_hits
                        >= self.config.tracking.confirmation_hits
                    ]
                    people_2d = visible_measured + visible_held
                    self.performance.tick("segmentation")
                    self._segmentation_ms = (time.perf_counter() - start) * 1000.0
                    confirmed_people = [
                        detection
                        for detection in measured_people
                        if detection.track_hits
                        >= self.config.tracking.confirmation_hits
                        and not detection.is_prediction
                        and detection.confidence >= self.config.segmentation.tracking_confidence
                    ]
                    observations, _ = extractor.extract(confirmed_people, cloud)
                    tracks = tracker_3d.update(
                        observations,
                        cloud.timestamp,
                    )
                    relationship_tracks = [
                        track
                        for track in tracks
                        if track.hit_count >= 1
                        and track.missing_count
                        <= self.config.tracking.obstacle_cloud_hold_updates
                    ]
                    published_obstacle_points = sum(
                        observation.point_count for observation in observations
                    )
                    if self.yolo_obstacle_pointcloud_publisher is not None:
                        try:
                            publish_observations = _observations_with_short_hold(
                                observations,
                                tracks,
                                observation_cache,
                                self.config.tracking.obstacle_cloud_hold_updates,
                                cloud.timestamp,
                                voxel_size=self.config.reconstruction.voxel_size,
                                max_center_step_m=(
                                    self.config.tracking.obstacle_center_max_step_m
                                ),
                            )
                            obstacle_cloud = _observation_pointcloud(
                                publish_observations,
                                cloud,
                                self.config.reconstruction.max_points,
                            )
                            published_obstacle_points = len(obstacle_cloud.points)
                            self.yolo_obstacle_pointcloud_publisher.publish(obstacle_cloud)
                        except Exception as publish_exc:
                            if "yolo_obstacle_pointcloud_topic" not in self._errors:
                                LOGGER.exception("YOLO obstacle point-cloud publication failed")
                                self._errors["yolo_obstacle_pointcloud_topic"] = str(publish_exc)
                    if self.arm_obstacle_relationship_publisher is not None:
                        try:
                            self.arm_obstacle_relationship_publisher.publish(
                                robot_arm,
                                relationship_tracks,
                                source_timestamp=cloud.timestamp,
                            )
                        except Exception as publish_exc:
                            if "arm_obstacle_relationship_topic" not in self._errors:
                                LOGGER.exception(
                                    "Arm-obstacle relationship publication failed"
                                )
                                self._errors["arm_obstacle_relationship_topic"] = str(
                                    publish_exc
                                )
                    diagnostic_now = time.perf_counter()
                    if diagnostic_now - last_yolo_diagnostics_log >= 2.0:
                        accepted_objects = {id(detection) for detection in inference}
                        LOGGER.info(
                            "YOLO3D diagnostics frame=%d raw_2d=%d accepted_2d=%d "
                            "measured=%d confirmed=%d depth_shape=%dx%d "
                            "observations_3d=%d live_points=%d published_points=%d",
                            cloud.frame_index,
                            len(raw_inference),
                            len(inference),
                            len(measured_people),
                            len(confirmed_people),
                            cloud.pointmap.shape[1],
                            cloud.pointmap.shape[0],
                            len(observations),
                            sum(observation.point_count for observation in observations),
                            published_obstacle_points,
                        )
                        if robot_arm is not None:
                            LOGGER.info(
                                "Robot arm center xyz=(%.3f, %.3f, %.3f)m "
                                "uv=(%.1f, %.1f) mask=%d depth_points=%d "
                                "confidence=%.2f held=%d",
                                *robot_arm.center_xyz,
                                *robot_arm.center_xy,
                                robot_arm.mask_pixels,
                                robot_arm.point_count,
                                robot_arm.confidence,
                                robot_arm.held_frames,
                            )
                        for detection in raw_inference[:8]:
                            LOGGER.info(
                                "YOLO2D class=%s confidence=%.3f bbox=(%.1f,%.1f,%.1f,%.1f) "
                                "mask_pixels=%d self_filter=%s track=%s hits=%d",
                                detection.class_name,
                                detection.confidence,
                                *detection.bbox_xyxy,
                                (
                                    int(np.count_nonzero(detection.mask))
                                    if detection.mask is not None
                                    else 0
                                ),
                                "accepted" if id(detection) in accepted_objects else "rejected",
                                detection.track_id,
                                detection.track_hits,
                            )
                        for diagnostic in extractor.last_diagnostics[:8]:
                            depth_range = (
                                "none"
                                if diagnostic.depth_median_m is None
                                else (
                                    f"{diagnostic.depth_min_m:.3f}/"
                                    f"{diagnostic.depth_median_m:.3f}/"
                                    f"{diagnostic.depth_max_m:.3f}m"
                                )
                            )
                            LOGGER.info(
                                "YOLO3D detection class=%s confidence=%.3f track=%s "
                                "bbox=%s mask=%d sampled=%d valid_depth=%d "
                                "depth_min/median/max=%s filtered=%d output=%d reason=%s",
                                diagnostic.class_name,
                                diagnostic.confidence,
                                diagnostic.track_id,
                                tuple(round(value, 1) for value in diagnostic.bbox_xyxy),
                                diagnostic.mask_pixels,
                                diagnostic.sampled_mask_pixels,
                                diagnostic.valid_depth_pixels,
                                depth_range,
                                diagnostic.filtered_points,
                                diagnostic.output_points,
                                diagnostic.reason,
                            )
                        if confirmed_people and not observations:
                            LOGGER.warning(
                                "YOLO detections were confirmed but produced no valid 3D "
                                "obstacle points; inspect the YOLO3D detection diagnostics above"
                            )
                        last_yolo_diagnostics_log = diagnostic_now
                else:
                    tracks = []
                    if self.arm_obstacle_relationship_publisher is not None:
                        self.arm_obstacle_relationship_publisher.publish(
                            None,
                            [],
                            source_timestamp=cloud.timestamp,
                        )
                handled_cloud_version = cloud_version
                with self._state_lock:
                    self._people_tracks = [
                        track
                        for track in tracks
                        if track.missing_count
                        <= self.config.tracking.visual_hold_updates
                    ]
                    self._people_detections = people_2d
                    self._robot_arm = robot_arm
                    self._people_detection_frame = source_frame
                    self._people_frame_index = cloud.frame_index
                    self._people_cloud_version = cloud_version
                    self._people_version += 1
            except Exception as exc:
                handled_cloud_version = cloud_version
                LOGGER.exception("3D person extraction failed")
                self._errors["people_runtime"] = str(exc)
                with self._state_lock:
                    self._people_tracks = []
                    self._people_detections = []
                    self._robot_arm = None
                    self._people_detection_frame = source_frame
                    self._people_frame_index = cloud.frame_index
                    self._people_cloud_version = cloud_version
                    self._people_version += 1
        self.segmentation.close()

    def _safety_worker(self) -> None:
        maximum_depth = (
            self.config.reconstruction.max_metric_depth_m
            or self.config.reconstruction.max_relative_depth
        )
        extractor = ObstacleExtractor3D(
            self.config.reconstruction.confidence_threshold,
            maximum_depth,
            self.config.reconstruction.voxel_size,
        )
        tracker = Tracker3D(self.config.tracking)
        danger = DangerZonePredictor(self.config.safety)
        planner = LocalSafetyPlanner(self.config.safety)
        decision = SafetyDecisionEngine(self.config.safety)
        ground_estimator = GroundPlaneEstimator(camera_height=self.config.camera_height)
        handled_cloud_version = -1
        reset_counter = self._reset_counter
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            wait = next_tick - time.perf_counter()
            if wait > 0 and self._stop_event.wait(wait):
                break
            next_tick = max(next_tick + 1.0 / max(self.config.safety.target_hz, 0.1), time.perf_counter())
            with self._state_lock:
                perception = self._perceptions[-1] if self._perceptions else None
                cloud = self._cloud
                cloud_version = self._cloud_version
            if perception is None:
                continue
            if reset_counter != self._reset_counter:
                tracker.reset()
                decision.reset()
                ground_estimator.reset()
                handled_cloud_version = -1
                reset_counter = self._reset_counter
            if cloud is not None and cloud_version != handled_cloud_version:
                with self._state_lock:
                    aligned = min(self._perceptions, key=lambda item: abs(item.frame.frame_index - cloud.frame_index))
                observations, assigned = extractor.extract(aligned.detections, cloud)
                # Unknown clustering is intentionally lower-rate than safety prediction.
                if cloud_version % 5 == 0:
                    observations.extend(extractor.find_unknown(cloud, assigned))
                tracks = tracker.update(observations, cloud.timestamp)
                ground = ground_estimator.estimate(cloud.points)
                handled_cloud_version = cloud_version
            else:
                tracks = tracker.predict_to(perception.frame.source_timestamp)
                ground = None
            zones = danger.predict(tracks)
            plan = planner.plan(tracks, zones)
            if cloud is not None:
                traversable = compute_traversable_region(cloud.points, ground, tracks, zones) if ground is not None else self._traversable
            else:
                traversable = self._traversable
            metric_valid = self.config.scale_mode == "rgbd" or (
                self.config.scale_mode == "calibrated"
                and (
                    self.config.manual_scale is not None
                    or (cloud is not None and cloud.metric_scale is not None)
                )
            )
            snapshot = decision.update(
                perception.frame.source_timestamp,
                perception.frame.frame_index,
                tracks,
                zones,
                plan,
                metric_valid=metric_valid,
                depth_valid=cloud is not None and cloud.valid,
                camera_motion_confidence=perception.camera_motion_confidence,
                model_ready=self._models_ready["depth"] or self._models_ready["st4rtrack"],
            )
            self.performance.tick("safety")
            self.performance.add_latency_ms((time.perf_counter() - perception.frame.capture_timestamp) * 1000.0)
            if self.session_logger is not None:
                perf = self.performance.snapshot(
                    self.capture_queue.dropped + self.reconstruction_queue.dropped,
                    max(self.capture_queue.qsize(), self.reconstruction_queue.qsize()),
                    self.config.video.queue_size,
                )
                self.session_logger.log(
                    snapshot,
                    self.adaptive.profile,
                    cloud.source if cloud is not None else self.config.reconstruction.depth_mode,
                    self.config.scale_mode,
                    perf,
                )
            with self._state_lock:
                self._safety = snapshot
                self._safety_cloud_version = handled_cloud_version
                self._traversable = traversable
                self._planner_result = plan

    def _video_renderer_worker(self) -> None:
        """Publish reconstruction-mode RGB frames independently of the 3D scene."""

        last_frame_index = -1
        next_update = time.perf_counter()
        interval = 1.0 / max(self.config.gui.video_fps, 0.1)
        while not self._stop_event.is_set():
            wait = next_update - time.perf_counter()
            if wait > 0 and self._stop_event.wait(wait):
                break
            # Clear before reading so a frame arriving after the snapshot will
            # wake us immediately. This avoids the old fixed 20 ms polling
            # penalty when capture and GUI deadlines narrowly missed.
            self._new_frame_event.clear()
            with self._state_lock:
                frame = self._latest_capture
            if frame is None or frame.frame_index == last_frame_index:
                self._new_frame_event.wait(timeout=min(interval, 0.05))
                continue
            image = frame.bgr.copy()
            if self.dashboard is not None:
                self.dashboard.update_video(image)
            if self.video_recorder is not None:
                self.video_recorder.write(image, frame.original_fps)
            self.performance.tick("display")
            last_frame_index = frame.frame_index
            next_update = max(next_update + interval, time.perf_counter())

    def _renderer_worker(self) -> None:
        last_displayed_frame_index = -1
        last_people_dashboard_version = -1
        last_scene_cloud_version = -1
        last_scene_people_version = -1
        last_scene_safety_id = 0
        last_video_update = 0.0
        last_status_update = 0.0
        while not self._stop_event.is_set():
            with self._state_lock:
                display_frame = self._latest_capture
                perception = self._perceptions[-1] if self._perceptions else None
                cloud = self._cloud
                cloud_version = self._cloud_version
                people = list(self._people_tracks)
                people_detections = list(self._people_detections)
                robot_arm = self._robot_arm
                people_detection_frame = self._people_detection_frame
                people_version = self._people_version
                people_cloud_version = self._people_cloud_version
                people_frame_index = self._people_frame_index
                safety = self._safety
                safety_cloud_version = self._safety_cloud_version
                traversable = self._traversable
                planner = self._planner_result
            now = time.perf_counter()
            aligned_people_ready = (
                not self.config.people_overlay
                or (
                    people_cloud_version >= cloud_version
                    and cloud is not None
                    and people_frame_index == cloud.frame_index
                )
            )
            scene_cloud_due = (
                self.scene is not None
                and cloud is not None
                and cloud_version != last_scene_cloud_version
            )
            scene_people_due = (
                self.scene is not None
                and self.config.mode == "reconstruction"
                and self.config.people_overlay
                and people_version != last_scene_people_version
                and aligned_people_ready
            )
            scene_safety_due = (
                self.scene is not None
                and self.config.mode == "safety"
                and safety is not None
                and id(safety) != last_scene_safety_id
            )
            people_dashboard_due = (
                self.dashboard is not None
                and people_detection_frame is not None
                and people_version != last_people_dashboard_version
            )
            # Reconstruction RGB has its own lightweight renderer; this worker
            # may spend time serializing the point cloud without slowing video.
            video_due = (
                self.config.mode != "reconstruction"
                and now - last_video_update >= 1.0 / max(self.config.gui.video_fps, 0.1)
            )
            status_due = now - last_status_update >= 0.5
            if display_frame is None:
                self._stop_event.wait(0.01)
                continue
            if not (
                video_due
                or status_due
                or scene_cloud_due
                or scene_people_due
                or scene_safety_due
                or people_dashboard_due
            ):
                self._stop_event.wait(0.005)
                continue
            level = safety.safety_state if safety is not None else SafetyLevel.DEGRADED
            dropped = self.capture_queue.dropped + self.reconstruction_queue.dropped
            queue_size = max(self.capture_queue.qsize(), self.reconstruction_queue.qsize())
            performance = self.performance.snapshot(dropped, queue_size, self.config.video.queue_size)
            profile = "4D VIEWER" if self.config.mode == "reconstruction" else self.adaptive.observe(performance)
            display_detections = (
                people_detections
                if self.config.mode == "reconstruction" and self.config.people_overlay
                else perception.detections if perception is not None else []
            )
            annotated = (
                display_frame.bgr.copy()
                if self.config.mode == "reconstruction"
                else draw_video_overlay(display_frame.bgr, display_detections, level, performance)
            )
            is_new_display_frame = video_due and display_frame.frame_index != last_displayed_frame_index
            if self.video_recorder is not None and is_new_display_frame:
                self.video_recorder.write(annotated, display_frame.original_fps)
            pipeline_snapshot = PipelineSnapshot(
                frame=display_frame,
                annotated_bgr=annotated,
                detections=display_detections,
                pointcloud=cloud,
                robot_arm=robot_arm,
                people=people,
                safety=safety,
                performance=performance,
                profile=profile,
                depth_mode=cloud.source if cloud else self.config.reconstruction.depth_mode,
                scale_mode=self.config.scale_mode,
                status={"models": dict(self._models_ready), "errors": self.errors},
            )
            self.gui_state.publish(pipeline_snapshot)
            if self.dashboard is not None:
                if video_due:
                    self.dashboard.update_video(annotated)
                if status_due:
                    self.dashboard.update_performance(performance)
                    if self.config.mode == "reconstruction":
                        self.dashboard.update_reconstruction_status(
                            pipeline_snapshot.depth_mode,
                            gpu_info(self.config.device).name,
                            self._models_ready["st4rtrack"] or self._models_ready["depth"],
                            self.errors,
                            len(people_detections),
                            len(people),
                            self._models_ready["segmentation"],
                            sum(track.missing_count > 0 for track in people),
                            metric_scale=cloud.metric_scale if cloud is not None else None,
                            reference_depth_m=cloud.reference_depth_m if cloud is not None else None,
                            observed_reference_depth=(
                                cloud.reference_observed_depth if cloud is not None else None
                            ),
                        )
                    else:
                        self.dashboard.update_status(
                            level,
                            profile,
                            pipeline_snapshot.depth_mode,
                            self.config.scale_mode,
                            gpu_info(self.config.device).name,
                            safety.recommended_action.value if safety else "WAIT",
                        )
                if people_dashboard_due:
                    self.dashboard.update_people_detections(
                        people_detection_frame,
                        people_detections,
                        robot_arm=robot_arm,
                        tracks=people,
                    )
                    last_people_dashboard_version = people_version
            if self.scene is not None:
                if scene_cloud_due and cloud is not None:
                    if self.config.mode == "reconstruction" and self.config.people_overlay:
                        # Never let slower YOLO/3D-person alignment freeze the
                        # primary point cloud. When YOLO for this depth frame is
                        # not ready yet, update only the persistent point-cloud
                        # buffers and keep the last confirmed obstacle handles.
                        # Clearing them here and rebuilding them milliseconds
                        # later was the main GUI center/box flicker.
                        if aligned_people_ready:
                            self.scene.update_aligned_frame(
                                cloud,
                                people,
                                yolo_count=len(people_detections),
                                robot_arm=robot_arm,
                            )
                        else:
                            self.scene.update_pointcloud(cloud)
                        if aligned_people_ready:
                            last_scene_people_version = people_version
                    else:
                        self.scene.update_pointcloud(cloud)
                    last_scene_cloud_version = cloud_version
                elif scene_people_due:
                    self.scene.update_people(
                        people_frame_index,
                        people,
                        yolo_count=len(people_detections),
                        robot_arm=robot_arm,
                    )
                    last_scene_people_version = people_version
                if scene_safety_due:
                    self.scene.update_obstacles(safety.tracks, safety.danger_zones)
                    self.scene.update_navigation(traversable, planner)
                    last_scene_safety_id = id(safety)
            if is_new_display_frame:
                self.performance.tick("display")
                last_displayed_frame_index = display_frame.frame_index
            if video_due:
                last_video_update = now
            if status_due:
                last_status_update = now
            reconstruction_unavailable = (
                not self._models_ready["depth"]
                and not self._models_ready["st4rtrack"]
                and "depth" in self._errors
            )
            if self.config.mode == "reconstruction":
                reconstruction_unavailable = reconstruction_unavailable or (
                    "st4rtrack" in self._errors and not self._models_ready["depth"]
                )
                final_people_ready = not self.config.people_overlay or people_cloud_version >= cloud_version
                final_3d_ready = reconstruction_unavailable or (
                    cloud is not None
                    and cloud.frame_index >= self._final_frame_index
                    and final_people_ready
                )
                if (
                    self._capture_complete.is_set()
                    and display_frame.frame_index >= self._final_frame_index
                    and final_3d_ready
                ):
                    self._source_done.set()
            else:
                final_3d_ready = reconstruction_unavailable or (
                    cloud is not None
                    and cloud.frame_index >= self._final_frame_index
                    and safety_cloud_version >= cloud_version
                )
                if (
                    self._capture_complete.is_set()
                    and perception is not None
                    and perception.frame.frame_index >= self._final_frame_index
                    and safety is not None
                    and final_3d_ready
                ):
                    self._source_done.set()
            if (
                self.config.mode == "safety"
                and not self._calibrated
                and self._segmentation_ms is not None
                and self._reconstruction_ms is not None
            ):
                self.adaptive.startup_calibration(self._segmentation_ms, self._reconstruction_ms)
                self._calibrated = True


def _make_cuda_stream(device: str):
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        return torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
    except Exception:
        return None
