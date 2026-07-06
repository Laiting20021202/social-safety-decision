from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from packages.common_models import (
    AgentClass,
    AnalysisPacket,
    AnalysisSystemStatus,
    FramePacket,
    MotionEstimate,
    Point2D,
    RoadSegmentationResult,
    RobotCorridor,
    TrackObservation,
    VQADirectionEstimate,
)
from packages.frame_sources import FrameSource
from packages.overlay_renderer import (
    approximate_bev_point,
    default_robot_corridor,
    dynamic_risk_zone,
    estimate_motion_from_history,
    fuse_direction,
    polygon_from_image_to_bev,
)
from packages.overlay_renderer.zone_store import ZoneStore
from services.dataset_service.bev_projection import (
    profile_for_frame,
    project_pixel_to_bev_normalized,
)
from services.dataset_service.lightweight_tracker import LightweightVisionTracker
from services.dataset_service.sam3_client import Sam3Client


class AnalysisBuilder:
    def __init__(
        self,
        frame_source: FrameSource,
        zone_store: ZoneStore,
        sam3_client: Sam3Client | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.zone_store = zone_store
        self.sam3_client = sam3_client or Sam3Client()
        self._sam3_cache: dict[tuple[str, int], dict[str, object] | None] = {}
        self._sam3_road_cache: dict[str, dict[str, object] | None] = {}
        self._sam3_blockers: dict[str, dict[str, str]] = {}
        self._active_scenario_id: str | None = None
        self._active_video_session_id: str | None = None
        self._active_video_fps: float = 5.0
        self.lightweight_tracker = LightweightVisionTracker(max_tracks=6)

    def reset_if_scenario_changed(self, scenario_id: str) -> None:
        if self._active_scenario_id in {None, scenario_id}:
            self._active_scenario_id = scenario_id
            return
        self.close_active_session()
        self.lightweight_tracker.reset()
        self._sam3_cache = {
            key: value for key, value in self._sam3_cache.items() if key[0] == scenario_id
        }
        self._active_scenario_id = scenario_id

    def close_active_session(self) -> dict[str, object] | None:
        session_id = self._active_video_session_id
        self._active_video_session_id = None
        self._active_scenario_id = None
        if not session_id or not self.sam3_client.configured:
            return None
        return self.sam3_client.close_session(session_id)

    def prepare_video_tracking(
        self,
        scenario_id: str,
        video_path: Path,
        prompt: str = "person",
        prompt_frame_index: int = 0,
        video_fps: float = 5.0,
        max_objects: int = 6,
        max_frame_num_to_track: int | None = None,
    ) -> dict[str, object]:
        if not self.sam3_client.configured:
            self._set_sam3_blocker(scenario_id, "video", "SAM3_SERVICE_URL is not configured.")
            return {
                "status": "unavailable",
                "scenario_id": scenario_id,
                "message": "SAM3_SERVICE_URL is not configured.",
            }
        self.reset_if_scenario_changed(scenario_id)
        if self._active_video_session_id is not None:
            self.close_active_session()
            self._active_scenario_id = scenario_id
        started = self.sam3_client.start_video_session(
            video_path,
            scenario_id=scenario_id,
            analysis_fps=video_fps,
            max_objects=max_objects,
        )
        if not started or started.get("source") == "sam3_unavailable":
            self._set_sam3_blocker(scenario_id, "video", _sam3_message(started))
            return {
                "status": "degraded",
                "scenario_id": scenario_id,
                "message": _sam3_message(started),
            }
        session_id = str(started["session_id"])
        self._active_scenario_id = scenario_id
        self._active_video_session_id = session_id
        self._active_video_fps = video_fps
        prompted = self.sam3_client.add_video_prompt(
            session_id=session_id,
            prompt=prompt,
            frame_index=prompt_frame_index,
        )
        if prompted and prompted.get("source") == "sam3":
            self._store_video_frame_result(
                scenario_id,
                prompted,
                video_fps,
                tracking_scope="prompt_frame",
            )
            prompt_timestamp_sec = max(0.0, float(prompt_frame_index) / max(video_fps, 1e-6))
            prompt_detection_count = _detection_count(prompted)
        else:
            message = _sam3_message(prompted)
            self._set_sam3_blocker(scenario_id, "video", message)
            self.sam3_client.close_session(session_id)
            self._active_video_session_id = None
            return {
                "status": "degraded",
                "scenario_id": scenario_id,
                "session_id": session_id,
                "message": message,
            }
        propagated = self.sam3_client.propagate_video(
            session_id=session_id,
            start_frame_index=prompt_frame_index,
            max_frame_num_to_track=max_frame_num_to_track,
            propagation_direction="forward",
            include_frames=True,
        )
        if not propagated or propagated.get("source") == "sam3_unavailable":
            self._set_sam3_blocker(scenario_id, "video", _sam3_message(propagated))
            return {
                "status": "degraded",
                "scenario_id": scenario_id,
                "session_id": session_id,
                "message": _sam3_message(propagated),
                "prompt_result_available": prompt_detection_count > 0,
                "prompt_detection_count": prompt_detection_count,
                "prompt_frame_index": prompt_frame_index,
                "prompt_frame_timestamp_sec": prompt_timestamp_sec,
                "cached_analysis_frames": self._cached_analysis_frame_count(scenario_id),
            }
        frames = propagated.get("frames")
        if isinstance(frames, list):
            for frame_result in frames:
                if isinstance(frame_result, dict):
                    self._store_video_frame_result(
                        scenario_id,
                        frame_result,
                        video_fps,
                        tracking_scope="propagated",
                    )
        self._clear_sam3_blocker(scenario_id, "video")
        return {
            "status": "ok",
            "scenario_id": scenario_id,
            "session_id": session_id,
            "prompt": prompt,
            "video_path": str(video_path),
            "tracking_fps": video_fps,
            "cached_analysis_frames": len(
                [key for key in self._sam3_cache if key[0] == scenario_id]
            ),
            "prompt_result_available": prompt_detection_count > 0,
            "prompt_detection_count": prompt_detection_count,
            "prompt_frame_index": prompt_frame_index,
            "prompt_frame_timestamp_sec": prompt_timestamp_sec,
            "sam3": propagated,
        }

    def prepare_road_segmentation(
        self,
        scenario_id: str,
        frame_index: int = 0,
        prompts: list[str] | None = None,
    ) -> dict[str, object]:
        prompts = prompts or ["road", "sidewalk", "walkable path"]
        if not self.sam3_client.configured:
            self._set_sam3_blocker(scenario_id, "road", "SAM3_SERVICE_URL is not configured.")
            return {
                "status": "unavailable",
                "scenario_id": scenario_id,
                "message": "SAM3_SERVICE_URL is not configured.",
            }
        self.reset_if_scenario_changed(scenario_id)
        image_path = self.frame_source.get_frame_image_path(scenario_id, frame_index)
        result = self.sam3_client.segment_image(image_path, prompts)
        if not result or result.get("source") == "sam3_unavailable":
            message = _sam3_message(result)
            self._set_sam3_blocker(scenario_id, "road", message)
            return {
                "status": "degraded",
                "scenario_id": scenario_id,
                "frame_index": frame_index,
                "message": message,
            }
        self._sam3_road_cache[scenario_id] = {
            **result,
            "source_frame_index": frame_index,
        }
        self._clear_sam3_blocker(scenario_id, "road")
        road_detection = _sam3_road_detection(result)
        return {
            "status": "ok" if road_detection is not None else "degraded",
            "scenario_id": scenario_id,
            "frame_index": frame_index,
            "detections": len(result.get("detections", []))
            if isinstance(result.get("detections"), list)
            else 0,
            "message": _sam3_message(result),
            "sam3": result,
        }

    def packet(
        self,
        scenario_id: str,
        video_timestamp_sec: float,
        prediction_horizon_sec: float = 3.0,
        vqa_update_interval_sec: float = 2.0,
    ) -> AnalysisPacket:
        scenario = self.frame_source.get_scenario(scenario_id)
        frame_index = _frame_index_at_time(
            video_timestamp_sec,
            scenario.frame_count,
            scenario.duration_sec,
        )
        frame = self.frame_source.get_frame(scenario_id, frame_index)
        camera_profile = profile_for_frame(frame)
        sam3_result = self._sam3_result(scenario_id, frame_index)
        sam3_blocker = self._sam3_blocker_message(scenario_id)
        road = self._road_result(scenario_id, frame_index, frame.timestamp_sec, sam3_result)
        robot_corridor = RobotCorridor(
            scenario_id=scenario_id,
            timestamp_sec=frame.timestamp_sec,
            polygon=default_robot_corridor(),
            origin=Point2D(x=0.5, y=1.0),
            heading_vector=Point2D(x=0.0, y=-1.0),
            confidence=0.45 if road.is_valid else 0.25,
            metadata={
                "mode": (
                    "SCAND estimated IPM BEV"
                    if camera_profile
                    else "Approximate BEV - RGB-only"
                ),
                "road_source": road.source,
                "camera_profile": camera_profile.name if camera_profile else "none",
            },
        )
        raw_tracks = self._track_observations(scenario_id, frame_index, sam3_result)
        tracks = [_track_with_bev_metadata(track, frame) for track in raw_tracks]
        propagated_tracking = _is_propagated_tracking_result(sam3_result)
        prompt_frame_preview = _is_prompt_frame_preview(sam3_result)
        history_by_track = (
            self._track_history(scenario_id, frame_index) if not prompt_frame_preview else {}
        )
        motions = [
            motion
            for track_id in sorted(history_by_track)
            if (motion := estimate_motion_from_history(history_by_track[track_id])) is not None
        ]
        motions = [self._fuse_motion(motion, None) for motion in motions]
        motion_by_id = {motion.track_id: motion for motion in motions}
        bev_tracks = [_image_track_to_bev(track, frame) for track in tracks]
        risk_zones = (
            [
                dynamic_risk_zone(
                    track=track,
                    motion=motion_by_id.get(track.track_id),
                    robot_corridor=robot_corridor.polygon,
                    prediction_horizon_sec=prediction_horizon_sec,
                )
                for track in bev_tracks
            ]
            if not prompt_frame_preview
            else []
        )
        analysis_timestamp_sec = frame.timestamp_sec
        delay_ms = max(0, int((video_timestamp_sec - analysis_timestamp_sec) * 1000))
        sam31_video_result = _is_sam31_video_result(sam3_result)
        lightweight_tracking = _is_lightweight_tracking_result(tracks)
        tracking_status: Literal["ok", "degraded", "unavailable"] = (
            "ok"
            if tracks and (propagated_tracking or not sam31_video_result)
            else "unavailable"
        )
        road_status: Literal["ok", "degraded", "unavailable"] = (
            "ok"
            if road.is_valid
            else ("degraded" if self._has_sam3_blocker(scenario_id, "road") else "unavailable")
        )
        vqa_last = (
            math.floor(video_timestamp_sec / vqa_update_interval_sec) * vqa_update_interval_sec
        )
        return AnalysisPacket(
            scenario_id=scenario_id,
            video_timestamp_sec=max(0.0, video_timestamp_sec),
            analysis_timestamp_sec=analysis_timestamp_sec,
            road=road,
            tracks=tracks,
            motions=motions,
            vqa_directions=[
                VQADirectionEstimate(
                    track_id=track.track_id,
                    direction_label="uncertain",
                    path_relation="uncertain",
                    confidence=0.0,
                    reason="Temporal VQA service is not loaded.",
                    updated_at_sec=vqa_last,
                    parse_valid=False,
                )
                for track in tracks
            ],
            risk_zones=risk_zones,
            robot_corridor=robot_corridor,
            system_status=AnalysisSystemStatus(
                tracking_fps=(
                    self._active_video_fps
                    if propagated_tracking and tracks
                    else (_scenario_fps(scenario) if lightweight_tracking else 0.0)
                ),
                vqa_update_interval_sec=vqa_update_interval_sec,
                analysis_delay_ms=delay_ms,
                analysis_age_ms=delay_ms,
                vqa_last_update_sec=vqa_last,
                tracking_status=tracking_status,
                road_status=road_status,
                vqa_status="unavailable",
                message=_status_message(
                    road_status,
                    tracking_status,
                    sam3_message=_sam3_message(sam3_result, fallback=sam3_blocker),
                    tracking_mode="lightweight_visual_tracker" if lightweight_tracking else None,
                ),
            ),
            metadata={
                "analysis_source": "dataset-service",
                "initializer": _initializer_name(tracks, sam3_result),
                "formal_model_output": _formal_model_output(sam3_result),
                "tracking_mode": (
                    "sam3.1_video_tracking"
                    if propagated_tracking
                    else (
                        "sam3_prompt_frame_preview"
                        if prompt_frame_preview
                        else (
                            "lightweight_visual_tracker"
                            if lightweight_tracking
                            else "unavailable"
                        )
                    )
                ),
                "sam3_message": _sam3_message(sam3_result, fallback=sam3_blocker),
                "sam3_blocker": sam3_blocker,
                "sam3_video_session_id": self._active_video_session_id,
                "sam3_precomputed": sam3_result is not None,
                "sam3_prompt_frame_preview": prompt_frame_preview,
                "sam3_cross_frame_tracking": propagated_tracking,
                "segmentation_status": (
                    "prompt_frame_preview"
                    if prompt_frame_preview
                    else (
                        "cross_frame_tracking"
                        if propagated_tracking
                        else (
                            "not_used_lightweight_boxes"
                            if lightweight_tracking
                            else "unavailable"
                        )
                    )
                ),
                "vqa_message": (
                    "Temporal VQA service is not loaded. Speed and direction are currently "
                    "from BEV geometry history; VQA is reserved for semantic confirmation."
                ),
                "bev_method": (
                    "RGB-only perspective approximation from image ground-contact points; "
                    "not metric depth."
                ),
            },
        )

    def _store_video_frame_result(
        self,
        scenario_id: str,
        frame_result: dict[str, object],
        video_fps: float,
        *,
        tracking_scope: Literal["prompt_frame", "propagated"],
    ) -> None:
        frame_index = frame_result.get("frame_index")
        if not isinstance(frame_index, int | float):
            return
        timestamp_sec = max(0.0, float(frame_index) / max(video_fps, 1e-6))
        scenario = self.frame_source.get_scenario(scenario_id)
        scenario_frame_index = _frame_index_at_time(
            timestamp_sec,
            scenario.frame_count,
            scenario.duration_sec,
        )
        enriched = {
            **frame_result,
            "source": "sam3",
            "tracking_scope": tracking_scope,
            "stable_tracking": tracking_scope == "propagated",
            "formal_model_output": tracking_scope == "propagated",
            "analysis_video_frame_index": int(frame_index),
            "analysis_video_fps": video_fps,
            "timestamp_sec": timestamp_sec,
        }
        self._sam3_cache[(scenario_id, scenario_frame_index)] = enriched

    def _road_result(
        self,
        scenario_id: str,
        frame_index: int,
        timestamp_sec: float,
        sam3_result: dict[str, object] | None,
    ) -> RoadSegmentationResult:
        sam3_road = _sam3_road_detection(sam3_result)
        if sam3_road is not None:
            polygon = _points_from_payload(sam3_road.get("mask_polygon"))
            mask_rle_payload = sam3_road.get("mask_rle")
            mask_rle = mask_rle_payload if isinstance(mask_rle_payload, dict) else None
            if polygon:
                return RoadSegmentationResult(
                    scenario_id=scenario_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    source="robopoint_sam3",
                    mask_rle=mask_rle,
                    polygon=polygon,
                    confidence=_float_value(sam3_road.get("confidence"), default=0.5),
                    prompt=str(sam3_road.get("label") or "road/walkable path"),
                    is_valid=True,
                    metadata={"source": "sam3", "formal_model_output": True},
                )
        zone = self.zone_store.load(self.frame_source.dataset_info().dataset_id, scenario_id)
        if zone and zone.polygon:
            return RoadSegmentationResult(
                scenario_id=scenario_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                source="manual_fallback",
                polygon=zone.polygon,
                confidence=zone.confidence,
                prompt=zone.prompt,
                is_valid=True,
                metadata={"zone_id": zone.zone_id, "name": zone.name},
            )
        if self._is_fixture_scenario(scenario_id):
            frame = self.frame_source.get_frame(scenario_id, frame_index)
            width = frame.image_width
            height = frame.image_height
            return RoadSegmentationResult(
                scenario_id=scenario_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                source="fixture_color_segmentation",
                polygon=[
                    Point2D(x=0, y=height * 0.695),
                    Point2D(x=width, y=height * 0.695),
                    Point2D(x=width, y=height),
                    Point2D(x=0, y=height),
                ],
                confidence=0.5,
                prompt="Fixture-only road color segmentation.",
                is_valid=True,
                metadata={"formal_model_output": False},
            )
        return RoadSegmentationResult(
            scenario_id=scenario_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            source="unavailable",
            prompt="Mark the walkable path in front of the robot.",
            is_valid=False,
        )

    def _track_observations(
        self,
        scenario_id: str,
        frame_index: int,
        sam3_result: dict[str, object] | None = None,
    ) -> list[TrackObservation]:
        sam3_tracks = _track_observations_from_sam3(
            self.frame_source.get_frame(scenario_id, frame_index),
            sam3_result,
        )
        if sam3_tracks:
            return sam3_tracks
        if not self._is_fixture_scenario(scenario_id):
            return self.lightweight_tracker.observations_until(
                scenario_id,
                frame_index,
                self.frame_source,
            )
        path = self.frame_source.get_frame_image_path(scenario_id, frame_index)
        detection = _fixture_person_detection(path)
        if detection is None:
            return []
        bbox, centroid, ground = detection
        return [
            TrackObservation(
                track_id=1,
                class_name="person",
                timestamp_sec=self.frame_source.get_frame(scenario_id, frame_index).timestamp_sec,
                frame_index=frame_index,
                mask_polygon=[
                    Point2D(x=bbox[0], y=bbox[1]),
                    Point2D(x=bbox[2], y=bbox[1]),
                    Point2D(x=bbox[2], y=bbox[3]),
                    Point2D(x=bbox[0], y=bbox[3]),
                ],
                bounding_box=bbox,
                centroid=centroid,
                centroid_image=centroid,
                ground_contact_point=ground,
                bottom_center=ground,
                confidence=0.72,
                track_age_sec=self.frame_source.get_frame(scenario_id, frame_index).timestamp_sec,
                metadata={
                    "initializer": "fixture_color_segmentation",
                    "formal_model_output": False,
                    "segmentation_note": "Only enabled for local test fixture frames.",
                },
            )
        ]

    def _track_history(
        self,
        scenario_id: str,
        frame_index: int,
        window: int = 5,
    ) -> dict[int, list[TrackObservation]]:
        scenario = self.frame_source.get_scenario(scenario_id)
        start = max(0, frame_index - window + 1)
        history: dict[int, list[TrackObservation]] = {}
        for index in range(start, min(frame_index + 1, scenario.frame_count)):
            frame = self.frame_source.get_frame(scenario_id, index)
            sam3_result = self._sam3_result(scenario_id, index)
            for observation in self._track_observations(scenario_id, index, sam3_result):
                history.setdefault(observation.track_id, []).append(
                    _image_track_to_bev(
                        observation,
                        frame=frame,
                    )
                )
        return history

    def _fuse_motion(
        self,
        motion: MotionEstimate,
        vqa: VQADirectionEstimate | None,
    ) -> MotionEstimate:
        label, confidence, _conflict = fuse_direction(
            motion.direction_label_geometry,
            vqa,
            motion.confidence,
        )
        return motion.model_copy(
            update={
                "direction_label_vqa": vqa.direction_label if vqa else "uncertain",
                "direction_label_fused": label,
                "confidence": confidence,
            }
        )

    def _is_fixture_scenario(self, scenario_id: str) -> bool:
        first_frame = self.frame_source.get_frame_image_path(scenario_id, 0)
        sidecar = first_frame.parent / "n_people.json"
        if not sidecar.exists():
            return False
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        n_people = data.get("n_people", 0)
        return isinstance(n_people, int | float) and n_people > 0

    def _sam3_result(self, scenario_id: str, frame_index: int) -> dict[str, object] | None:
        key = (scenario_id, frame_index)
        video_result = self._sam3_cache.get(key)
        road_result = self._sam3_road_cache.get(scenario_id)
        if video_result is None:
            return road_result
        if road_result is None:
            return video_result
        merged_detections: list[object] = []
        for result in (road_result, video_result):
            detections = result.get("detections") if isinstance(result, dict) else None
            if isinstance(detections, list):
                merged_detections.extend(detections)
        return {
            **video_result,
            "detections": merged_detections,
            "road_source": road_result.get("task"),
        }

    def _set_sam3_blocker(self, scenario_id: str, kind: str, message: str) -> None:
        self._sam3_blockers.setdefault(scenario_id, {})[kind] = message

    def _clear_sam3_blocker(self, scenario_id: str, kind: str) -> None:
        blockers = self._sam3_blockers.get(scenario_id)
        if not blockers:
            return
        blockers.pop(kind, None)
        if not blockers:
            self._sam3_blockers.pop(scenario_id, None)

    def _has_sam3_blocker(self, scenario_id: str, kind: str) -> bool:
        return kind in self._sam3_blockers.get(scenario_id, {})

    def _sam3_blocker_message(self, scenario_id: str) -> str | None:
        blockers = self._sam3_blockers.get(scenario_id)
        if not blockers:
            return None
        return "; ".join(blockers[key] for key in sorted(blockers))

    def _cached_analysis_frame_count(self, scenario_id: str) -> int:
        return len([key for key in self._sam3_cache if key[0] == scenario_id])


def road_polygon_to_bev(
    road: RoadSegmentationResult,
    image_width: float,
    image_height: float,
) -> list[Point2D]:
    return polygon_from_image_to_bev(road.polygon, image_width, image_height)


def _track_observations_from_sam3(
    frame: FramePacket,
    sam3_result: dict[str, object] | None,
) -> list[TrackObservation]:
    if not sam3_result or sam3_result.get("source") != "sam3":
        return []
    detections = sam3_result.get("detections")
    if not isinstance(detections, list):
        return []
    tracks: list[TrackObservation] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "unknown")
        if class_name not in {"person", "bicycle", "motorcycle", "car", "bus", "truck"}:
            continue
        track_id = _track_id_from_detection(detection)
        if track_id is None:
            continue
        transform = _sam3_detection_transform(frame, detection)
        polygon = _points_from_payload(detection.get("mask_polygon"))
        polygon = [_transform_point(point, transform) for point in polygon]
        bbox = _bbox_from_payload(detection.get("bounding_box"))
        bbox = _transform_bbox(bbox, transform)
        ground_payload = detection.get("ground_contact_point")
        ground = _point_from_payload(ground_payload)
        ground = _transform_point(ground, transform) if ground is not None else None
        if ground is None and bbox is not None:
            ground = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=bbox[3])
        centroid = _point_from_payload(detection.get("centroid"))
        centroid = _transform_point(centroid, transform) if centroid is not None else None
        if centroid is None:
            centroid = _polygon_centroid(polygon) if polygon else None
        if centroid is None and bbox is not None:
            centroid = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=(bbox[1] + bbox[3]) / 2.0)
        if centroid is None or ground is None:
            continue
        mask_rle = (
            detection.get("mask_rle") if isinstance(detection.get("mask_rle"), dict) else None
        )
        tracks.append(
            TrackObservation(
                track_id=track_id,
                class_name=cast(AgentClass, class_name),
                timestamp_sec=frame.timestamp_sec,
                frame_index=frame.frame_index,
                mask_rle=mask_rle,
                mask_polygon=polygon,
                bounding_box=bbox,
                centroid=centroid,
                centroid_image=centroid,
                ground_contact_point=ground,
                bottom_center=ground,
                confidence=_float_value(detection.get("confidence"), default=0.5),
                track_age_sec=0.0,
                metadata={
                    "initializer": "sam3",
                    "formal_model_output": _formal_model_output(sam3_result),
                    "label": detection.get("label"),
                    "official_object_id": track_id,
                    "tracking_scope": str(sam3_result.get("tracking_scope") or "unknown"),
                    "stable_track_id": _is_propagated_tracking_result(sam3_result),
                    "segmentation_type": detection.get("segmentation_type"),
                    "mask_format": detection.get("mask_format"),
                    "coordinate_transform": transform,
                },
            )
        )
    return tracks


def _detection_count(result: dict[str, object] | None) -> int:
    if not isinstance(result, dict):
        return 0
    detections = result.get("detections")
    return len(detections) if isinstance(detections, list) else 0


def _sam3_detection_transform(
    frame: FramePacket,
    detection: dict[str, object],
) -> dict[str, float | bool | str]:
    source_size = _mask_size_from_rle(detection.get("mask_rle"))
    if source_size is None:
        return {
            "applied": False,
            "source": "identity",
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
    source_height, source_width = source_size
    if source_width <= 0 or source_height <= 0:
        return {
            "applied": False,
            "source": "identity",
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
    if frame.image_width == source_width and frame.image_height == source_height:
        return {
            "applied": False,
            "source": "identity",
            "source_width": float(source_width),
            "source_height": float(source_height),
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
    source_aspect = source_width / source_height
    frame_aspect = frame.image_width / max(frame.image_height, 1)
    if frame_aspect < source_aspect:
        target_width = float(frame.image_width)
        target_height = min(float(frame.image_height), target_width / source_aspect)
        offset_x = 0.0
        offset_y = 0.0
        source = "top_crop_width_fit"
    else:
        target_height = float(frame.image_height)
        target_width = min(float(frame.image_width), target_height * source_aspect)
        offset_x = 0.0
        offset_y = 0.0
        source = "full_height_fit"
    return {
        "applied": True,
        "source": source,
        "source_width": float(source_width),
        "source_height": float(source_height),
        "target_width": target_width,
        "target_height": target_height,
        "scale_x": target_width / source_width,
        "scale_y": target_height / source_height,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def _mask_size_from_rle(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    size = value.get("size")
    if not isinstance(size, list | tuple) or len(size) != 2:
        return None
    height, width = size
    if not isinstance(height, int | float) or not isinstance(width, int | float):
        return None
    return int(height), int(width)


def _transform_point(
    point: Point2D,
    transform: dict[str, float | bool | str],
) -> Point2D:
    return Point2D(
        x=float(transform["offset_x"]) + point.x * float(transform["scale_x"]),
        y=float(transform["offset_y"]) + point.y * float(transform["scale_y"]),
    )


def _transform_bbox(
    bbox: tuple[float, float, float, float] | None,
    transform: dict[str, float | bool | str],
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    top_left = _transform_point(Point2D(x=bbox[0], y=bbox[1]), transform)
    bottom_right = _transform_point(Point2D(x=bbox[2], y=bbox[3]), transform)
    return (top_left.x, top_left.y, bottom_right.x, bottom_right.y)


def _sam3_road_detection(sam3_result: dict[str, object] | None) -> dict[str, object] | None:
    if not sam3_result or sam3_result.get("source") != "sam3":
        return None
    detections = sam3_result.get("detections")
    if not isinstance(detections, list):
        return None
    road_classes = {"road", "sidewalk", "corridor", "walkable path", "traversable ground"}
    candidates = [
        detection
        for detection in detections
        if isinstance(detection, dict) and str(detection.get("class_name")) in road_classes
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: _float_value(item.get("confidence"), default=0.0))


def _track_id_from_detection(detection: dict[str, object]) -> int | None:
    for key in ("track_id", "object_id"):
        value = detection.get(key)
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value >= 0 and value.is_integer():
            return int(value)
    return None


def _points_from_payload(value: object) -> list[Point2D]:
    if not isinstance(value, list):
        return []
    points: list[Point2D] = []
    for item in value:
        point = _point_from_payload(item)
        if point is not None:
            points.append(point)
    return points


def _point_from_payload(value: object) -> Point2D | None:
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    y = value.get("y")
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return None
    return Point2D(x=float(x), y=float(y))


def _bbox_from_payload(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    if not all(isinstance(item, int | float) for item in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _polygon_centroid(polygon: list[Point2D]) -> Point2D | None:
    if not polygon:
        return None
    return Point2D(
        x=sum(point.x for point in polygon) / len(polygon),
        y=sum(point.y for point in polygon) / len(polygon),
    )


def _float_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return default


def _initializer_name(
    tracks: list[TrackObservation],
    sam3_result: dict[str, object] | None,
) -> str:
    if sam3_result and sam3_result.get("source") == "sam3":
        return "sam3"
    if tracks:
        return str(tracks[0].metadata.get("initializer") or "unknown")
    return "unavailable"


def _formal_model_output(sam3_result: dict[str, object] | None) -> bool:
    if not sam3_result or sam3_result.get("source") != "sam3":
        return False
    if sam3_result.get("task") == "sam3.1_video_tracking":
        return _is_propagated_tracking_result(sam3_result)
    return sam3_result.get("task") == "image_concept_segmentation"


def _is_sam31_video_result(sam3_result: dict[str, object] | None) -> bool:
    return bool(
        sam3_result
        and sam3_result.get("source") == "sam3"
        and sam3_result.get("task") == "sam3.1_video_tracking"
    )


def _is_prompt_frame_preview(sam3_result: dict[str, object] | None) -> bool:
    return (
        _is_sam31_video_result(sam3_result)
        and sam3_result.get("tracking_scope") == "prompt_frame"
    )


def _is_propagated_tracking_result(sam3_result: dict[str, object] | None) -> bool:
    return _is_sam31_video_result(sam3_result) and sam3_result.get("tracking_scope") == "propagated"


def _is_lightweight_tracking_result(tracks: list[TrackObservation]) -> bool:
    return bool(
        tracks
        and str(tracks[0].metadata.get("initializer") or "") == "lightweight_visual_tracker"
    )


def _scenario_fps(scenario: object) -> float:
    frame_count = getattr(scenario, "frame_count", 0)
    duration_sec = getattr(scenario, "duration_sec", 0.0)
    if isinstance(frame_count, int | float) and isinstance(duration_sec, int | float):
        if frame_count > 1 and duration_sec > 1e-6:
            return min(10.0, max(0.0, float(frame_count - 1) / float(duration_sec)))
    return 0.0


def _sam3_message(sam3_result: dict[str, object] | None, fallback: str | None = None) -> str:
    if sam3_result is None:
        return fallback or "No precomputed SAM 3 tracking result for this timestamp."
    if sam3_result.get("source") == "sam3":
        if sam3_result.get("task") == "sam3.1_video_tracking":
            if _is_prompt_frame_preview(sam3_result):
                return (
                    "SAM3 Prompt-Frame Segmentation Preview. "
                    "Segmentation available, cross-frame tracking unavailable."
                )
            if _is_propagated_tracking_result(sam3_result):
                return "SAM 3.1 precomputed cross-frame video tracking available."
            return "SAM 3.1 video tracking unavailable."
        return "SAM3 segmentation available."
    return str(sam3_result.get("message") or "SAM3 unavailable.")


def _status_message(
    road_status: str,
    tracking_status: str,
    *,
    sam3_message: str | None = None,
    tracking_mode: str | None = None,
) -> str:
    messages = []
    if road_status != "ok":
        messages.append("Road segmentation unavailable")
    prompt_preview = bool(sam3_message and "Prompt-Frame Segmentation Preview" in sam3_message)
    if prompt_preview:
        messages.append("Segmentation available, cross-frame tracking unavailable.")
    elif tracking_mode == "lightweight_visual_tracker":
        messages.append("Lightweight visual tracker active; boxes only, no SAM mask.")
    elif tracking_status != "ok":
        messages.append("SAM 3 tracking unavailable")
    if tracking_mode == "lightweight_visual_tracker":
        if sam3_message and ("401" in sam3_message or "gated" in sam3_message.lower()):
            messages.append(sam3_message)
    elif sam3_message and "unavailable" not in sam3_message.lower():
        messages.append(sam3_message)
    elif sam3_message and ("401" in sam3_message or "gated" in sam3_message.lower()):
        messages.append(sam3_message)
    return "; ".join(messages)


def _frame_index_at_time(timestamp_sec: float, frame_count: int, duration_sec: float) -> int:
    if frame_count <= 1:
        return 0
    interval = duration_sec / (frame_count - 1) if duration_sec > 0 else 1.0 / 25.0
    return max(0, min(frame_count - 1, int(timestamp_sec / max(interval, 1e-6))))


def _fixture_person_detection(
    path: Path,
) -> tuple[tuple[float, float, float, float], Point2D, Point2D] | None:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    rows = np.arange(array.shape[0])[:, None]
    mask = (red < 70) & (green > 70) & (green < 140) & (blue > 60) & (blue < 130)
    mask &= rows > array.shape[0] * 0.2
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    centroid = Point2D(x=float(xs.mean()), y=float(ys.mean()))
    bottom_band = xs[ys >= y2 - 2]
    ground_x = float(bottom_band.mean()) if len(bottom_band) else (x1 + x2) / 2.0
    ground = Point2D(x=ground_x, y=y2)
    return (x1, y1, x2, y2), centroid, ground


def _track_with_bev_metadata(
    observation: TrackObservation,
    frame: FramePacket,
) -> TrackObservation:
    bev_ground, metadata = _bev_ground_point_for_observation(observation, frame)
    return observation.model_copy(
        update={
            "metadata": {
                **observation.metadata,
                "bev_ground_point": bev_ground.model_dump(mode="json"),
                "bev_projection": metadata,
            }
        }
    )


def _image_track_to_bev(
    observation: TrackObservation,
    frame: FramePacket,
) -> TrackObservation:
    bev_ground, _metadata = _bev_ground_point_for_observation(observation, frame)
    return observation.model_copy(
        update={
            "centroid": bev_ground,
            "centroid_image": observation.centroid,
            "ground_contact_point": bev_ground,
            "bottom_center": bev_ground,
        }
    )


def _bev_ground_point_for_observation(
    observation: TrackObservation,
    frame: FramePacket,
) -> tuple[Point2D, dict[str, object]]:
    ground = observation.ground_contact_point or observation.bottom_center or observation.centroid
    camera_profile = profile_for_frame(frame)
    if camera_profile is not None:
        projected = project_pixel_to_bev_normalized(
            ground,
            image_width=frame.image_width,
            image_height=frame.image_height,
            profile=camera_profile,
        )
        if projected is not None:
            return projected, {
                "mode": "ground_plane_ipm",
                "profile": camera_profile.name,
                "camera_height_m": camera_profile.camera_height_m,
                "pitch_down_deg": camera_profile.pitch_down_deg,
                "lateral_range_m": [
                    camera_profile.lateral_min_m,
                    camera_profile.lateral_max_m,
                ],
                "forward_range_m": [
                    camera_profile.forward_min_m,
                    camera_profile.forward_max_m,
                ],
            }
    return approximate_bev_point(
        ground,
        frame.image_width,
        _effective_visual_height(frame.image_width, frame.image_height),
    ), {
        "mode": "rgb_only_approximate",
        "profile": "none",
    }


def _effective_visual_height(image_width: float, image_height: float) -> float:
    if image_height > image_width * 0.85:
        return max(1.0, min(image_height, image_width * 9.0 / 16.0))
    return max(1.0, image_height)
