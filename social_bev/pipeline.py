from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from social_bev.bev_map import BEVMapBuilder
from social_bev.config import load_classes, load_config, load_detection_config
from social_bev.detection import ObjectDetector
from social_bev.homography import estimate_ground_contact, image_points_to_bev_pixels, load_calibration
from social_bev.segmentation import WalkableSegmenter
from social_bev.tracking import MultiObjectTracker
from social_bev.types import Calibration, Detection, ObstacleRegion, PipelineResult, Track
from social_bev.unknown_obstacles import UnknownObstacleExtractor
from social_bev.utils import apply_cpu_thread_settings
from social_bev.visualization import compose_visualization


LOGGER = logging.getLogger(__name__)


class SocialNavigationPipeline:
    """End-to-end CPU monocular RGB perception pipeline for social navigation BEV."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        calibration_path: str | Path | None = None,
        class_config: dict[str, list[str]] | None = None,
        detection_class_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        runtime = self.config.get("runtime", {})
        if str(runtime.get("device", "cpu")).lower() != "cpu":
            LOGGER.warning("Only CPU execution is supported; overriding runtime.device to cpu")
            runtime["device"] = "cpu"
        apply_cpu_thread_settings(int(runtime.get("cpu_threads", 0) or 0))

        self.input_width = int(runtime.get("input_width", 640))
        self.input_height = int(runtime.get("input_height", 360))
        self.segmentation_interval = max(1, int(runtime.get("segmentation_interval", 2)))
        self.detection_interval = max(1, int(runtime.get("detection_interval", 2)))
        self.reuse_previous_results = bool(runtime.get("reuse_previous_results", True))
        self.frame_index = 0
        self._fps_sum = 0.0

        self.class_config = class_config or load_classes()
        self.detection_class_config = detection_class_config or load_detection_config()
        self.segmenter = WalkableSegmenter(self.config.get("segmentation", {}), self.class_config)
        self.detector = ObjectDetector(self.config.get("detection", {}), self.detection_class_config)
        self.tracker = MultiObjectTracker(self.config.get("tracking", {}))
        self.unknown_extractor = UnknownObstacleExtractor(self.config.get("unknown_obstacles", {}))
        frame_shape = (self.input_height, self.input_width, 3)
        self.calibration: Calibration = load_calibration(calibration_path, self.config.get("bev", {}), frame_shape)
        self.bev_builder = BEVMapBuilder(self.config.get("bev", {}), self.config.get("social_zone", {}))
        self._last_segmentation_mask: np.ndarray | None = None
        self._last_detections: list[Detection] = []
        self._last_backend_label = "seg:pending det:pending"

    def process_frame(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        total_start = time.perf_counter()
        processing_ms: dict[str, float] = {}
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        segmentation_run = self._should_run(self.frame_index, self.segmentation_interval, self._last_segmentation_mask)
        if segmentation_run:
            seg_result = self.segmenter.predict(resized)
            walkable_mask = seg_result.mask
            self._last_segmentation_mask = walkable_mask
            processing_ms["segmentation"] = seg_result.processing_ms
            segmentation_backend = seg_result.backend
        elif self.reuse_previous_results and self._last_segmentation_mask is not None:
            walkable_mask = self._last_segmentation_mask.copy()
            processing_ms["segmentation"] = 0.0
            segmentation_backend = "reused"
        else:
            walkable_mask = np.zeros(resized.shape[:2], dtype=bool)
            processing_ms["segmentation"] = 0.0
            segmentation_backend = "empty"

        detection_run = self._should_run(self.frame_index, self.detection_interval, self._last_detections)
        if detection_run:
            detection_start = time.perf_counter()
            detections = self.detector.predict(resized)
            processing_ms["detection"] = self.detector.last_processing_ms or (time.perf_counter() - detection_start) * 1000.0
            self._last_detections = detections
            tracker_predict_only = False
            detection_backend = self.detector.backend
        elif self.reuse_previous_results:
            detections = list(self._last_detections)
            processing_ms["detection"] = 0.0
            tracker_predict_only = True
            detection_backend = "reused"
        else:
            detections = []
            processing_ms["detection"] = 0.0
            tracker_predict_only = False
            detection_backend = "empty"

        track_start = time.perf_counter()
        tracks = self.tracker.update(detections, timestamp, predict_only=tracker_predict_only)
        self._project_tracks(tracks, walkable_mask)
        tracks = self.tracker.tracks()
        processing_ms["tracking"] = (time.perf_counter() - track_start) * 1000.0

        unknown_start = time.perf_counter()
        unknown_obstacles = self.unknown_extractor.extract(walkable_mask, detections, resized.shape)
        processing_ms["unknown_obstacles"] = (time.perf_counter() - unknown_start) * 1000.0

        bev_start = time.perf_counter()
        bev = self.bev_builder.build(
            walkable_mask=walkable_mask,
            detections=detections,
            unknown_obstacles=unknown_obstacles,
            tracks=tracks,
            calibration=self.calibration,
            frame_shape=resized.shape,
        )
        processing_ms["bev"] = (time.perf_counter() - bev_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        processing_ms["total"] = total_ms
        fps = 1000.0 / total_ms if total_ms > 1e-6 else 0.0
        self._fps_sum += fps
        average_fps = self._fps_sum / max(1, self.frame_index + 1)
        self._last_backend_label = f"seg:{segmentation_backend} det:{detection_backend}"
        annotated, visualization = compose_visualization(
            frame=resized,
            walkable_mask=walkable_mask,
            detections=detections,
            tracks=tracks,
            unknown_obstacles=unknown_obstacles,
            bev=bev,
            processing_ms=processing_ms,
            fps=fps,
            average_fps=average_fps,
            frame_index=self.frame_index,
            backend_label=self._last_backend_label,
            config=self.config.get("visualization", {}),
        )
        result = PipelineResult(
            frame_index=self.frame_index,
            timestamp=float(timestamp),
            annotated_frame=annotated,
            visualization=visualization,
            walkable_mask=walkable_mask,
            detections=detections,
            tracks=tracks,
            unknown_obstacles=unknown_obstacles,
            bev=bev,
            processing_ms=processing_ms,
            fps=fps,
            average_fps=average_fps,
        )
        self.frame_index += 1
        return result

    def _should_run(self, frame_index: int, interval: int, previous: object | None) -> bool:
        if previous is None:
            return True
        return frame_index % interval == 0

    def _project_tracks(self, tracks: list[Track], walkable_mask: np.ndarray) -> None:
        for track in tracks:
            ground_point, confidence = estimate_ground_contact(track.bbox, walkable_mask)
            try:
                _, bev_points = image_points_to_bev_pixels(self.calibration, [ground_point])
                bev_position = (float(bev_points[0, 0]), float(bev_points[0, 1]))
            except Exception:
                bev_position = None
                confidence = 0.0
            self.tracker.set_track_projection(track.track_id, ground_point, bev_position, confidence)

