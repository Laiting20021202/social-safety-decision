from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from social_bev.types import BEVResult, Detection, ObstacleRegion, Track
from social_bev.utils import letterbox, overlay_mask


PERSON_COLOR = (60, 220, 60)
KNOWN_COLOR = (0, 165, 255)
UNKNOWN_COLOR = (80, 80, 240)
TRACK_COLOR = (255, 210, 60)


def draw_front_view(frame: np.ndarray, detections: list[Detection], tracks: list[Track]) -> np.ndarray:
    image = frame.copy()
    for detection in detections:
        color = PERSON_COLOR if detection.category == "person" else KNOWN_COLOR
        _draw_box(image, detection.bbox, color, f"{detection.class_name} {detection.confidence:.2f}")
    for track in tracks:
        _draw_box(image, track.bbox, TRACK_COLOR, f"ID {track.track_id}")
        if track.image_ground_point is not None:
            point = (int(round(track.image_ground_point[0])), int(round(track.image_ground_point[1])))
            cv2.circle(image, point, 5, TRACK_COLOR, -1)
    return image


def draw_obstacle_contacts(
    frame: np.ndarray,
    detections: list[Detection],
    unknown_obstacles: list[ObstacleRegion],
    tracks: list[Track],
) -> np.ndarray:
    image = frame.copy()
    for detection in detections:
        if detection.category == "known_obstacle":
            x1, _, x2, y2 = detection.bbox
            pts = [(x1, y2), ((x1 + x2) * 0.5, y2), (x2, y2)]
            for point in pts:
                cv2.circle(image, (int(point[0]), int(point[1])), 5, KNOWN_COLOR, -1)
            cv2.line(image, (int(x1), int(y2)), (int(x2), int(y2)), KNOWN_COLOR, 2)
    for obstacle in unknown_obstacles:
        cv2.drawContours(image, [obstacle.contour], -1, UNKNOWN_COLOR, 2)
        for point in obstacle.ground_points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, UNKNOWN_COLOR, -1)
        x1, y1, _, _ = obstacle.bbox
        cv2.putText(image, "unknown RGB estimate", (int(x1), max(18, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UNKNOWN_COLOR, 1, cv2.LINE_AA)
    for track in tracks:
        if track.image_ground_point is not None:
            point = (int(track.image_ground_point[0]), int(track.image_ground_point[1]))
            cv2.circle(image, point, 6, TRACK_COLOR, -1)
    return image


def compose_visualization(
    frame: np.ndarray,
    walkable_mask: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    unknown_obstacles: list[ObstacleRegion],
    bev: BEVResult,
    processing_ms: dict[str, float],
    fps: float,
    average_fps: float,
    frame_index: int,
    backend_label: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    front = draw_front_view(frame, detections, tracks)
    mask_overlay = overlay_mask(frame, walkable_mask)
    contacts = draw_obstacle_contacts(frame, detections, unknown_obstacles, tracks)
    bev_image = bev.image.copy()

    person_count = sum(1 for d in detections if d.category == "person")
    known_count = sum(1 for d in detections if d.category == "known_obstacle")
    unknown_count = len(unknown_obstacles)
    common_lines = [
        f"Frame {frame_index}",
        f"People {person_count}  Known {known_count}  Unknown {unknown_count}",
        f"{bev.label}",
        f"FPS {fps:.2f}  AVG {average_fps:.2f}",
        f"CPU backend {backend_label}",
    ]
    timing_lines = [
        f"Seg {processing_ms.get('segmentation', 0.0):.1f} ms",
        f"Det {processing_ms.get('detection', 0.0):.1f} ms",
        f"Track {processing_ms.get('tracking', 0.0):.1f} ms",
        f"BEV {processing_ms.get('bev', 0.0):.1f} ms",
        f"Total {processing_ms.get('total', 0.0):.1f} ms",
    ]
    _put_lines(front, common_lines, (12, 22))
    _put_lines(mask_overlay, timing_lines, (12, 22))
    _put_lines(contacts, ["Obstacle / ground contact", "Unknown shown as RGB estimate"], (12, 22))

    output_width = int(config.get("output_width", 1280))
    gap = int(config.get("panel_gap", 8))
    panel_w = max(240, (output_width - gap) // 2)
    source_h, source_w = frame.shape[:2]
    panel_h = max(180, int(round(panel_w * source_h / max(1, source_w))))
    canvas_w = panel_w * 2 + gap
    canvas_h = panel_h * 2 + gap
    canvas = np.full((canvas_h, canvas_w, 3), (18, 18, 18), dtype=np.uint8)

    panels = [
        letterbox(front, (panel_w, panel_h)),
        letterbox(mask_overlay, (panel_w, panel_h)),
        letterbox(contacts, (panel_w, panel_h)),
        letterbox(bev_image, (panel_w, panel_h)),
    ]
    positions = [(0, 0), (panel_w + gap, 0), (0, panel_h + gap), (panel_w + gap, panel_h + gap)]
    for panel, (x, y) in zip(panels, positions):
        canvas[y : y + panel_h, x : x + panel_w] = panel
    return front, canvas


def _draw_box(image: np.ndarray, bbox: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    _put_label(image, label, (x1, max(0, y1 - 8)), color)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(th + baseline + 2, y)
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), (15, 15, 15), -1)
    cv2.putText(image, text, (x + 2, y - 3), font, scale, color, thickness, cv2.LINE_AA)


def _put_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    line_h = 19
    width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        width = max(width, tw)
    cv2.rectangle(image, (x - 6, y - 17), (x + width + 10, y + line_h * len(lines)), (15, 15, 15), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (x, y + i * line_h), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)

