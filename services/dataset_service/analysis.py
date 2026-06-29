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
from services.dataset_service.sam3_client import Sam3Client


class AnalysisBuilder:
    def __init__(self, frame_source: FrameSource, zone_store: ZoneStore) -> None:
        self.frame_source = frame_source
        self.zone_store = zone_store
        self.sam3_client = Sam3Client()
        self._sam3_cache: dict[tuple[str, int], dict[str, object] | None] = {}

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
        sam3_result = self._sam3_result(scenario_id, frame_index)
        road = self._road_result(scenario_id, frame_index, frame.timestamp_sec, sam3_result)
        robot_corridor = RobotCorridor(
            scenario_id=scenario_id,
            timestamp_sec=frame.timestamp_sec,
            polygon=default_robot_corridor(),
            origin=Point2D(x=0.5, y=1.0),
            heading_vector=Point2D(x=0.0, y=-1.0),
            confidence=0.45 if road.is_valid else 0.25,
            metadata={
                "mode": "Approximate BEV - RGB-only",
                "road_source": road.source,
            },
        )
        tracks = self._track_observations(scenario_id, frame_index, sam3_result)
        history_by_track = self._track_history(scenario_id, frame_index)
        motions = [
            motion
            for track_id in sorted(history_by_track)
            if (motion := estimate_motion_from_history(history_by_track[track_id])) is not None
        ]
        motions = [self._fuse_motion(motion, None) for motion in motions]
        motion_by_id = {motion.track_id: motion for motion in motions}
        bev_tracks = [
            _image_track_to_bev(track, frame.image_width, frame.image_height) for track in tracks
        ]
        risk_zones = [
            dynamic_risk_zone(
                track=track,
                motion=motion_by_id.get(track.track_id),
                robot_corridor=robot_corridor.polygon,
                prediction_horizon_sec=prediction_horizon_sec,
            )
            for track in bev_tracks
        ]
        analysis_timestamp_sec = frame.timestamp_sec
        delay_ms = max(0, int((video_timestamp_sec - analysis_timestamp_sec) * 1000))
        tracking_status: Literal["ok", "degraded", "unavailable"] = (
            "ok" if tracks else ("degraded" if self.sam3_client.configured else "unavailable")
        )
        road_status: Literal["ok", "degraded", "unavailable"] = (
            "ok" if road.is_valid else "unavailable"
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
                tracking_fps=10.0 if tracks else 0.0,
                vqa_update_interval_sec=vqa_update_interval_sec,
                analysis_delay_ms=delay_ms,
                analysis_age_ms=delay_ms,
                vqa_last_update_sec=vqa_last,
                tracking_status=tracking_status,
                road_status=road_status,
                vqa_status="unavailable",
                message=_status_message(road_status, tracking_status),
            ),
            metadata={
                "analysis_source": "dataset-service",
                "initializer": _initializer_name(tracks, sam3_result),
                "formal_model_output": _formal_model_output(sam3_result),
                "sam3_message": _sam3_message(sam3_result),
            },
        )

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
            if polygon:
                return RoadSegmentationResult(
                    scenario_id=scenario_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    source="robopoint_sam3",
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
            return []
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
                        image_width=frame.image_width,
                        image_height=frame.image_height,
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
        if key in self._sam3_cache:
            return self._sam3_cache[key]
        if not self.sam3_client.configured:
            self._sam3_cache[key] = None
            return None
        image_path = self.frame_source.get_frame_image_path(scenario_id, frame_index)
        try:
            result = self.sam3_client.segment_image(
                image_path,
                prompts=[
                    "person",
                    "bicycle",
                    "motorcycle",
                    "car",
                    "bus",
                    "truck",
                    "road",
                    "sidewalk",
                    "corridor",
                    "walkable path",
                    "traversable ground",
                ],
            )
        except Exception as exc:
            result = {"source": "sam3_unavailable", "detections": [], "message": str(exc)}
        self._sam3_cache[key] = result
        return result


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
    track_id = 1
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "unknown")
        if class_name not in {"person", "bicycle", "motorcycle", "car", "bus", "truck"}:
            continue
        polygon = _points_from_payload(detection.get("mask_polygon"))
        bbox = _bbox_from_payload(detection.get("bounding_box"))
        ground_payload = detection.get("ground_contact_point")
        ground = _point_from_payload(ground_payload)
        if ground is None and bbox is not None:
            ground = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=bbox[3])
        centroid = _polygon_centroid(polygon) if polygon else None
        if centroid is None and bbox is not None:
            centroid = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=(bbox[1] + bbox[3]) / 2.0)
        if centroid is None or ground is None:
            continue
        tracks.append(
            TrackObservation(
                track_id=track_id,
                class_name=cast(AgentClass, class_name),
                timestamp_sec=frame.timestamp_sec,
                frame_index=frame.frame_index,
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
                    "formal_model_output": True,
                    "label": detection.get("label"),
                },
            )
        )
        track_id += 1
    return tracks


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
    return bool(sam3_result and sam3_result.get("source") == "sam3")


def _sam3_message(sam3_result: dict[str, object] | None) -> str:
    if sam3_result is None:
        return "SAM3_SERVICE_URL is not configured."
    if sam3_result.get("source") == "sam3":
        return "SAM3 segmentation available."
    return str(sam3_result.get("message") or "SAM3 unavailable.")


def _status_message(road_status: str, tracking_status: str) -> str:
    messages = []
    if road_status != "ok":
        messages.append("Road segmentation unavailable")
    if tracking_status != "ok":
        messages.append("SAM 3 tracking unavailable")
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


def _image_track_to_bev(
    observation: TrackObservation,
    image_width: float,
    image_height: float,
) -> TrackObservation:
    ground = observation.ground_contact_point or observation.bottom_center or observation.centroid
    bev_ground = approximate_bev_point(ground, image_width, image_height)
    return observation.model_copy(
        update={
            "centroid": bev_ground,
            "centroid_image": observation.centroid,
            "ground_contact_point": bev_ground,
            "bottom_center": bev_ground,
        }
    )
