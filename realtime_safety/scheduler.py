from __future__ import annotations

import logging
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
from realtime_safety.pipeline.safety_decision import SafetyDecisionEngine
from realtime_safety.pipeline.segmentation import SegmentationBackend, create_segmentation_backend
from realtime_safety.pipeline.st4rtrack_adapter import St4RTrackAdapter
from realtime_safety.pipeline.tracker_2d import StableTracker2D
from realtime_safety.pipeline.tracker_3d import Tracker3D
from realtime_safety.pipeline.traversable_region import TraversableRegion, compute_traversable_region
from realtime_safety.pipeline.video_source import PlaybackState, VideoSource
from realtime_safety.types import (
    Detection2D,
    FramePacket,
    PipelineSnapshot,
    PointCloudFrame,
    RecommendedAction,
    SafetyLevel,
    Track3DState,
)
from realtime_safety.utils.gpu import gpu_info, release_gpu_memory

LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        self.config = config
        self.segmentation = segmentation_backend
        if self.segmentation is None and (config.mode == "safety" or config.people_overlay):
            self.segmentation = create_segmentation_backend(config.segmentation, config.device)
        self.depth = depth_backend or MonocularDepthBackend(config.reconstruction, config.device)
        self.st4r = St4RTrackAdapter(config.reconstruction, config.device)
        self.dashboard = dashboard
        self.scene = scene
        self.session_logger = session_logger
        self.video_recorder = video_recorder
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
        self._stop_event.clear()
        if self.config.mode == "reconstruction":
            targets_list = [
                ("capture-worker", self._capture_worker),
                ("3d-reconstruction-worker", self._reconstruction_worker),
            ]
            if self.config.people_overlay:
                targets_list.append(("people-3d-worker", self._people_worker))
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
        if command == "start" and value not in (None, ""):
            source: str | int = int(value) if str(value).isdigit() else str(value)
            self.start_source(source)
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
        with self._source_lock:
            if self._source:
                self._source.close()
        for thread in self._threads:
            thread.join(timeout=10.0)
        self._threads.clear()
        self._running = False
        release_gpu_memory()
        if self.session_logger is not None:
            self.session_logger.close()
        if self.video_recorder is not None:
            self.video_recorder.close()

    def _capture_worker(self) -> None:
        next_deadline = time.perf_counter()
        last_source_identity = None
        last_frame_index = -1
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
            wait = next_deadline - time.perf_counter()
            if wait > 0 and self._stop_event.wait(wait):
                break
            frame = source.read()
            if frame is None:
                continue
            if frame.frame_index <= last_frame_index:
                self._reset_runtime_state()
            last_frame_index = frame.frame_index
            if self.config.mode == "safety":
                self.capture_queue.put_latest(frame)
            self.reconstruction_queue.put_latest(frame)
            self.performance.tick("input")
            with self._state_lock:
                self._latest_capture = frame
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
        extractor = ObstacleExtractor3D(
            self.config.reconstruction.confidence_threshold,
            self.config.reconstruction.max_relative_depth,
            self.config.reconstruction.voxel_size,
            minimum_points=20,
        )
        tracker_2d = StableTracker2D(self.config.tracking)
        tracker_3d = Tracker3D(self.config.tracking)
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
                handled_cloud_version = -1
                reset_counter = self._reset_counter
            try:
                people_2d: list[Detection2D] = []
                if self._models_ready["segmentation"]:
                    start = time.perf_counter()
                    def run_inference() -> list[Detection2D]:
                        infer_people = getattr(self.segmentation, "infer_people", None)
                        if callable(infer_people):
                            return infer_people(source_frame)
                        return [
                            detection
                            for detection in self.segmentation.infer(source_frame)
                            if detection.class_name == "person"
                        ]

                    if runtime_stream is None:
                        with self._gpu_lock:
                            inference = run_inference()
                    else:
                        import torch

                        with torch.cuda.stream(runtime_stream):
                            inference = run_inference()
                    people_2d = tracker_2d.update(
                        inference,
                        source_frame.source_timestamp,
                    )
                    self.performance.tick("segmentation")
                    self._segmentation_ms = (time.perf_counter() - start) * 1000.0
                    confirmed_people = [
                        detection
                        for detection in people_2d
                        if detection.track_hits >= 2 and detection.confidence >= self.config.segmentation.confidence
                    ]
                    observations, _ = extractor.extract(confirmed_people, cloud)
                    tracks = tracker_3d.update(observations, cloud.timestamp)
                else:
                    tracks = []
                handled_cloud_version = cloud_version
                with self._state_lock:
                    self._people_tracks = [
                        track
                        for track in tracks
                        if track.class_name == "person"
                        and track.missing_count <= self.config.tracking.visual_hold_updates
                    ]
                    self._people_detections = people_2d
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
                    self._people_detection_frame = source_frame
                    self._people_frame_index = cloud.frame_index
                    self._people_cloud_version = cloud_version
                    self._people_version += 1
        self.segmentation.close()

    def _safety_worker(self) -> None:
        extractor = ObstacleExtractor3D(
            self.config.reconstruction.confidence_threshold,
            self.config.reconstruction.max_relative_depth,
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
                self.config.scale_mode == "calibrated" and self.config.manual_scale is not None
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
                and (self.config.mode != "reconstruction" or aligned_people_ready)
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
            video_due = now - last_video_update >= 1.0 / 24.0
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
                    self.dashboard.update_people_detections(people_detection_frame, people_detections)
                    last_people_dashboard_version = people_version
            if self.scene is not None:
                if scene_cloud_due and cloud is not None:
                    if self.config.mode == "reconstruction" and self.config.people_overlay:
                        self.scene.update_aligned_frame(cloud, people, yolo_count=len(people_detections))
                        last_scene_people_version = people_version
                    else:
                        self.scene.update_pointcloud(cloud)
                    last_scene_cloud_version = cloud_version
                elif scene_people_due:
                    self.scene.update_people(people_frame_index, people, yolo_count=len(people_detections))
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
