from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from social_bev.types import Detection


LOGGER = logging.getLogger(__name__)


class ObjectDetector:
    """YOLO11 CPU detector for people and navigation obstacles."""

    def __init__(self, config: dict[str, Any], class_config: dict[str, Any]) -> None:
        self.config = config
        self.class_config = class_config
        self.backend = str(config.get("backend", "openvino")).lower()
        self.model_path = str(config.get("model", "models/yolo11n_openvino_model"))
        self.fallback_model = str(config.get("fallback_model", "yolo11n.pt"))
        self.input_size = int(config.get("input_size", class_config.get("input_size", 416)))
        self.confidence = float(config.get("confidence", class_config.get("confidence", 0.35)))
        self.iou_threshold = float(config.get("iou_threshold", class_config.get("iou_threshold", 0.45)))
        self.minimum_box_area = float(config.get("minimum_box_area", class_config.get("minimum_box_area", 120)))
        self.allow_empty_on_error = bool(config.get("allow_empty_on_error", True))
        self.allow_hog_fallback = bool(config.get("allow_hog_fallback", class_config.get("allow_hog_fallback", True)))
        self.person_classes = set(class_config.get("person_classes", ["person"]))
        self.obstacle_classes = set(class_config.get("obstacle_classes", []))
        self._model: Any | None = None
        self._hog: cv2.HOGDescriptor | None = None
        self._names: dict[int, str] = {}
        self._load_attempted = False
        self.last_processing_ms = 0.0

    def predict(self, frame: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        try:
            if not self._load_attempted:
                self._load_model()
            if self._model is None:
                raise RuntimeError("YOLO model is not loaded")
            results = self._model.predict(
                frame,
                imgsz=self.input_size,
                conf=self.confidence,
                iou=self.iou_threshold,
                device="cpu",
                verbose=False,
            )
            detections = self._parse_results(results)
        except Exception as exc:
            if not self.allow_empty_on_error:
                raise
            if self.allow_hog_fallback:
                LOGGER.warning("YOLO unavailable; using OpenCV HOG CPU person detector: %s", exc)
                detections = self._predict_hog(frame)
            else:
                LOGGER.warning("Object detection unavailable; returning empty detections: %s", exc)
                detections = []
        self.last_processing_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def _load_model(self) -> None:
        self._load_attempted = True
        from ultralytics import YOLO

        chosen_model = self._choose_model_path()
        LOGGER.info("Loading YOLO detector on CPU: %s", chosen_model)
        self._model = YOLO(chosen_model)
        names = getattr(self._model, "names", {})
        self._names = {int(k): str(v) for k, v in dict(names).items()}

    def _choose_model_path(self) -> str:
        if self.backend == "openvino" and Path(self.model_path).exists():
            return self.model_path
        if self.backend == "openvino":
            LOGGER.warning("OpenVINO YOLO model not found at %s; using %s", self.model_path, self.fallback_model)
        return self.fallback_model

    def _parse_results(self, results: Any) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", self._names)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                confidence = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item())
                class_name = str(names.get(class_id, self._names.get(class_id, str(class_id))))
                category = self._category_for(class_name)
                if category == "ignored":
                    continue
                detection = Detection(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    category=category,
                )
                if detection.area < self.minimum_box_area:
                    continue
                detections.append(detection)
        return detections

    def _category_for(self, class_name: str) -> str:
        if class_name in self.person_classes:
            return "person"
        if class_name in self.obstacle_classes:
            return "known_obstacle"
        return "ignored"

    def _predict_hog(self, frame: np.ndarray) -> list[Detection]:
        if self._hog is None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        scale = 1.0
        image = frame
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / float(max(h, w))
            image = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        rects, weights = self._hog.detectMultiScale(image, 0.0, (8, 8), (8, 8), 1.05, 2)
        detections: list[Detection] = []
        inv_scale = 1.0 / max(scale, 1e-6)
        for rect, weight in zip(rects, weights):
            x, y, bw, bh = [float(v) * inv_scale for v in rect]
            confidence = float(1.0 / (1.0 + np.exp(-float(weight))))
            detection = Detection(
                bbox=(x, y, x + bw, y + bh),
                confidence=confidence,
                class_id=0,
                class_name="person",
                category="person",
            )
            if detection.area >= self.minimum_box_area:
                detections.append(detection)
        return _nms_detections(detections, self.iou_threshold)


def _nms_detections(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not detections:
        return []
    boxes = np.asarray([d.bbox for d in detections], dtype=np.float32)
    scores = np.asarray([d.confidence for d in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_rest - inter, 1e-6)
        order = rest[iou <= iou_threshold]
    return [detections[i] for i in keep]
