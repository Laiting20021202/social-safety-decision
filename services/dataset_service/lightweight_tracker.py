from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from packages.common_models import AgentClass, Point2D, TrackObservation
from packages.frame_sources import FrameSource


@dataclass(frozen=True)
class _Detection:
    bbox: tuple[float, float, float, float]
    centroid: Point2D
    ground: Point2D
    class_name: AgentClass
    confidence: float


@dataclass
class _ActiveTrack:
    track_id: int
    class_name: AgentClass
    ground: Point2D
    bbox: tuple[float, float, float, float]
    first_timestamp_sec: float
    last_timestamp_sec: float
    lost_count: int = 0


@dataclass
class _ScenarioState:
    next_track_id: int = 1
    last_frame_index: int = -1
    active_tracks: dict[int, _ActiveTrack] = field(default_factory=dict)
    frame_observations: dict[int, list[TrackObservation]] = field(default_factory=dict)


class LightweightVisionTracker:
    """CPU-only visual proposal tracker used when SAM 3.1 tracking is not available.

    This is intentionally not a segmentation model. It extracts upright foreground proposals,
    associates them over nearby frames, and exposes bounding boxes plus ground-contact points
    so the BEV safety map can remain interactive while heavier models are unavailable.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        processing_width: int = 360,
        max_tracks: int = 6,
        max_gap: int = 8,
        max_backfill_frames: int = 30,
    ) -> None:
        self.enabled = enabled
        self.processing_width = processing_width
        self.max_tracks = max_tracks
        self.max_gap = max_gap
        self.max_backfill_frames = max_backfill_frames
        self._states: dict[str, _ScenarioState] = {}

    def reset(self, scenario_id: str | None = None) -> None:
        if scenario_id is None:
            self._states.clear()
            return
        self._states.pop(scenario_id, None)

    def observations_until(
        self,
        scenario_id: str,
        frame_index: int,
        frame_source: FrameSource,
    ) -> list[TrackObservation]:
        if not self.enabled:
            return []
        state = self._states.setdefault(scenario_id, _ScenarioState())
        if frame_index in state.frame_observations:
            return state.frame_observations[frame_index]

        if state.last_frame_index < 0 or frame_index < state.last_frame_index:
            start = max(0, frame_index - min(frame_index, self.max_backfill_frames - 1))
            state.active_tracks.clear()
            state.frame_observations.clear()
            state.last_frame_index = start - 1
        elif frame_index - state.last_frame_index > self.max_backfill_frames:
            start = max(0, frame_index - self.max_backfill_frames + 1)
            state.active_tracks.clear()
            state.last_frame_index = start - 1
        else:
            start = state.last_frame_index + 1

        scenario = frame_source.get_scenario(scenario_id)
        stop = min(frame_index, max(0, scenario.frame_count - 1))
        for index in range(start, stop + 1):
            frame = frame_source.get_frame(scenario_id, index)
            image_path = frame_source.get_frame_image_path(scenario_id, index)
            detections = self._detect(image_path)
            state.frame_observations[index] = self._associate(
                state,
                detections,
                frame.timestamp_sec,
                index,
            )
            state.last_frame_index = index
        return state.frame_observations.get(frame_index, [])

    def _associate(
        self,
        state: _ScenarioState,
        detections: list[_Detection],
        timestamp_sec: float,
        frame_index: int,
    ) -> list[TrackObservation]:
        observations: list[TrackObservation] = []
        assigned_tracks: set[int] = set()

        for detection in detections[: self.max_tracks]:
            match_id = self._best_track_match(state.active_tracks, assigned_tracks, detection)
            if match_id is None:
                match_id = state.next_track_id
                state.next_track_id += 1
                first_timestamp = timestamp_sec
            else:
                first_timestamp = state.active_tracks[match_id].first_timestamp_sec
            assigned_tracks.add(match_id)
            active = _ActiveTrack(
                track_id=match_id,
                class_name=detection.class_name,
                ground=detection.ground,
                bbox=detection.bbox,
                first_timestamp_sec=first_timestamp,
                last_timestamp_sec=timestamp_sec,
            )
            state.active_tracks[match_id] = active
            observations.append(
                TrackObservation(
                    track_id=match_id,
                    class_name=detection.class_name,
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bounding_box=detection.bbox,
                    centroid=detection.centroid,
                    centroid_image=detection.centroid,
                    ground_contact_point=detection.ground,
                    bottom_center=detection.ground,
                    confidence=detection.confidence,
                    track_age_sec=max(0.0, timestamp_sec - first_timestamp),
                    metadata={
                        "initializer": "lightweight_visual_tracker",
                        "formal_model_output": False,
                        "stable_track_id": True,
                        "track_id_source": "lightweight_nearest_neighbor_tracker",
                        "segmentation_type": "none",
                        "detection_shape": "bounding_box",
                        "source": "cpu_visual_proposal",
                    },
                )
            )

        for track_id, active in list(state.active_tracks.items()):
            if track_id in assigned_tracks:
                continue
            active.lost_count += 1
            if active.lost_count > self.max_gap:
                state.active_tracks.pop(track_id, None)
        return observations

    @staticmethod
    def _best_track_match(
        active_tracks: dict[int, _ActiveTrack],
        assigned_tracks: set[int],
        detection: _Detection,
    ) -> int | None:
        best_id: int | None = None
        best_score = float("inf")
        for track_id, track in active_tracks.items():
            if track_id in assigned_tracks:
                continue
            dx = detection.ground.x - track.ground.x
            dy = detection.ground.y - track.ground.y
            distance = (dx * dx + dy * dy) ** 0.5
            box_iou = _bbox_iou(detection.bbox, track.bbox)
            gate = 120.0 + track.lost_count * 45.0
            if detection.class_name != track.class_name and box_iou < 0.1:
                continue
            if distance > gate and box_iou < 0.08:
                continue
            score = distance - box_iou * 100.0 + track.lost_count * 20.0
            if score < best_score:
                best_score = score
                best_id = track_id
        return best_id

    def _detect(self, image_path: Path) -> list[_Detection]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_height = _visual_crop_height(width, height)
            image = image.crop((0, 0, width, crop_height))
            scale = min(1.0, self.processing_width / max(width, 1))
            small_size = (max(1, int(width * scale)), max(1, int(crop_height * scale)))
            if scale < 1.0:
                image = image.resize(small_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)

        mask = _foreground_mask(array)
        components = _connected_components(mask)
        detections: list[_Detection] = []
        small_height, small_width = mask.shape
        inv_scale = 1.0 / max(scale, 1e-6)
        for component in components:
            x1, y1, x2, y2, area = component
            box_width = max(1, x2 - x1 + 1)
            box_height = max(1, y2 - y1 + 1)
            if not _component_looks_useful(
                x1,
                y1,
                x2,
                y2,
                area,
                image_width=small_width,
                image_height=small_height,
            ):
                continue
            density = area / max(1, box_width * box_height)
            confidence = _component_confidence(
                box_width=box_width,
                box_height=box_height,
                density=density,
                y2=y2,
                image_height=small_height,
            )
            bbox = (
                float(max(0, x1 - 1) * inv_scale),
                float(max(0, y1 - 1) * inv_scale),
                float(min(small_width - 1, x2 + 1) * inv_scale),
                float(min(small_height - 1, y2 + 1) * inv_scale),
            )
            centroid = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=(bbox[1] + bbox[3]) / 2.0)
            ground = Point2D(x=(bbox[0] + bbox[2]) / 2.0, y=bbox[3])
            aspect = box_height / max(box_width, 1)
            class_name: AgentClass = "person" if aspect >= 1.15 else "unknown"
            detections.append(
                _Detection(
                    bbox=bbox,
                    centroid=centroid,
                    ground=ground,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        detections.sort(
            key=lambda item: (
                item.confidence,
                item.ground.y,
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            ),
            reverse=True,
        )
        return _non_max_suppress(detections, limit=max(1, self.max_tracks * 2))


def _visual_crop_height(width: int, height: int) -> int:
    if height > width * 0.85:
        return max(1, min(height, int(round(width * 9 / 16))))
    return height


def _foreground_mask(array: np.ndarray) -> np.ndarray:
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    gray = red * 0.299 + green * 0.587 + blue * 0.114
    max_channel = array.max(axis=2)
    min_channel = array.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gradient = np.maximum(grad_x, grad_y)
    rows = np.arange(array.shape[0], dtype=np.float32)[:, None]
    lower_scene = rows > array.shape[0] * 0.12
    not_sky = ~((blue > red + 18) & (blue > green + 12) & (gray > 125))
    chroma_or_dark = ((saturation > 0.18) & (gray < 238)) | (gray < 95)
    edge_object = (gradient > 34) & (gray < 245)
    mask = lower_scene & not_sky & (chroma_or_dark | edge_object)
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return _erode(_dilate(mask, iterations=1), iterations=1)


def _dilate(mask: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    result = mask
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def _erode(mask: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    result = mask
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return result


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for start_y in range(height):
        xs = np.where(mask[start_y] & ~visited[start_y])[0]
        for start_x in xs.tolist():
            if visited[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            x1 = x2 = start_x
            y1 = y2 = start_y
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            if area >= 12:
                components.append((x1, y1, x2, y2, area))
    return components


def _component_looks_useful(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    area: int,
    *,
    image_width: int,
    image_height: int,
) -> bool:
    box_width = max(1, x2 - x1 + 1)
    box_height = max(1, y2 - y1 + 1)
    aspect = box_height / box_width
    density = area / max(1, box_width * box_height)
    if box_height < max(16, image_height * 0.075):
        return False
    if box_width < 4:
        return False
    if area < 25:
        return False
    if box_width > image_width * 0.34 or box_height > image_height * 0.86:
        return False
    if aspect < 0.65 or aspect > 7.5:
        return False
    if density < 0.08 or density > 1.01:
        return False
    if y2 < image_height * 0.25:
        return False
    return True


def _component_confidence(
    *,
    box_width: int,
    box_height: int,
    density: float,
    y2: int,
    image_height: int,
) -> float:
    aspect = box_height / max(box_width, 1)
    aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 3.0)
    density_score = max(0.0, 1.0 - abs(density - 0.42) / 0.42)
    size_score = min(1.0, box_height / max(1.0, image_height * 0.28))
    ground_score = min(1.0, y2 / max(1.0, image_height * 0.85))
    score = (
        0.22
        + 0.32 * aspect_score
        + 0.18 * density_score
        + 0.18 * size_score
        + 0.1 * ground_score
    )
    return max(0.25, min(0.82, score))


def _non_max_suppress(detections: list[_Detection], *, limit: int) -> list[_Detection]:
    kept: list[_Detection] = []
    for detection in detections:
        if any(_bbox_iou(detection.bbox, existing.bbox) > 0.35 for existing in kept):
            continue
        kept.append(detection)
        if len(kept) >= limit:
            break
    return kept


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-6)
