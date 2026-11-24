from __future__ import annotations

import cv2
import numpy as np

from realtime_safety.types import Detection2D, PerformanceSnapshot, SafetyLevel


LEVEL_COLORS = {
    SafetyLevel.SAFE: (70, 210, 80),
    SafetyLevel.CAUTION: (0, 210, 255),
    SafetyLevel.WARNING: (0, 120, 255),
    SafetyLevel.STOP: (0, 0, 255),
    SafetyLevel.DEGRADED: (170, 80, 170),
}


def draw_video_overlay(
    bgr: np.ndarray,
    detections: list[Detection2D],
    safety_level: SafetyLevel = SafetyLevel.DEGRADED,
    performance: PerformanceSnapshot | None = None,
    show_masks: bool = True,
) -> np.ndarray:
    canvas = bgr.copy()
    overlay = canvas.copy()
    for detection in detections:
        track_id = detection.track_id or 0
        color = tuple(int(v) for v in _track_color(track_id))
        if show_masks and detection.mask is not None and detection.mask.shape == canvas.shape[:2]:
            overlay[detection.mask] = 0.55 * overlay[detection.mask] + 0.45 * np.asarray(color)
        x1, y1, x2, y2 = detection.bbox_xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"#{track_id} {detection.class_name} {detection.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        speed = float(np.linalg.norm(detection.velocity_xy))
        if speed > 2.0:
            end = tuple(np.rint(detection.centroid_xy + detection.velocity_xy * 0.15).astype(int))
            start = tuple(np.rint(detection.centroid_xy).astype(int))
            cv2.arrowedLine(canvas, start, end, color, 2, cv2.LINE_AA, tipLength=0.2)
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
    status_color = LEVEL_COLORS[safety_level]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(canvas, safety_level.value, (12, 29), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    if performance:
        metrics = (
            f"IN {performance.input_fps:.1f} | DISP {performance.display_fps:.1f} | "
            f"SEG {performance.segmentation_fps:.1f} | 3D {performance.reconstruction_fps:.1f} | "
            f"SAFE {performance.safety_fps:.1f} | P95 {performance.p95_latency_ms:.0f} ms"
        )
        cv2.putText(canvas, metrics, (145, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _track_color(track_id: int) -> np.ndarray:
    return np.array(
        [
            50 + (track_id * 97) % 205,
            50 + (track_id * 57) % 205,
            50 + (track_id * 137) % 205,
        ],
        dtype=np.uint8,
    )
