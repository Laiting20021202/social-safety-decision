from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from social_bev.types import SegmentationResult
from social_bev.utils import clean_binary_mask, keep_bottom_center_component


LOGGER = logging.getLogger(__name__)


def normalize_label(label: str) -> str:
    return label.lower().replace("_", " ").replace("-", " ").strip()


def merge_walkable_classes(
    label_map: np.ndarray,
    id2label: dict[int | str, str],
    class_config: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Merge ADE20K labels into a configurable walkable mask."""

    walkable = {normalize_label(v) for v in class_config.get("walkable_classes", [])}
    walkable.update(normalize_label(v) for v in class_config.get("optional_walkable_classes", []))
    blocked = {normalize_label(v) for v in class_config.get("blocked_classes", [])}
    mask = np.zeros(label_map.shape, dtype=bool)
    labels_present: list[str] = []
    for raw_id, raw_name in id2label.items():
        class_id = int(raw_id)
        name = normalize_label(str(raw_name))
        if np.any(label_map == class_id):
            labels_present.append(name)
        if name in walkable and name not in blocked:
            mask |= label_map == class_id
    return mask, sorted(set(labels_present))


class WalkableSegmenter:
    """CPU semantic segmenter with Torch/OpenVINO backends and bounded mask smoothing."""

    def __init__(self, config: dict[str, Any], class_config: dict[str, list[str]]) -> None:
        self.config = config
        self.class_config = class_config
        self.backend = str(config.get("backend", "torch")).lower()
        self.model_name = str(config.get("model", "nvidia/segformer-b0-finetuned-ade-512-512"))
        self.openvino_model = str(config.get("openvino_model", ""))
        self.input_size = int(config.get("input_size", 384))
        self.temporal_smoothing = float(config.get("temporal_smoothing", 0.65))
        self.minimum_component_area = int(config.get("minimum_component_area", 500))
        self.morphology_kernel = int(config.get("morphology_kernel", 5))
        self.allow_classical_fallback = bool(config.get("allow_classical_fallback", True))
        self._processor: Any | None = None
        self._model: Any | None = None
        self._compiled_model: Any | None = None
        self._ov_output: Any | None = None
        self._id2label: dict[int, str] = {}
        self._load_attempted = False
        self._fallback_reason: str | None = None
        self._ema: np.ndarray | None = None

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        start = time.perf_counter()
        raw_labels: np.ndarray | None = None
        confidence: np.ndarray | None = None
        labels_present: list[str] = []
        backend_used = self.backend
        try:
            if not self._load_attempted:
                self._load_backend()
            if self._fallback_reason:
                raise RuntimeError(self._fallback_reason)
            raw_labels, confidence, labels_present = self._predict_model(frame)
            mask, labels_present = merge_walkable_classes(raw_labels, self._id2label, self.class_config)
        except Exception as exc:
            if not self.allow_classical_fallback:
                LOGGER.warning("Segmentation failed; returning empty walkable mask: %s", exc)
                mask = np.zeros(frame.shape[:2], dtype=bool)
                backend_used = "unavailable"
            else:
                LOGGER.warning("Segmentation backend unavailable; using RGB heuristic: %s", exc)
                mask, confidence = self._classical_rgb_walkable(frame)
                raw_labels = None
                labels_present = ["rgb heuristic ground"]
                backend_used = "rgb_heuristic"

        mask = self._postprocess(mask, frame.shape[:2])
        elapsed = (time.perf_counter() - start) * 1000.0
        return SegmentationResult(
            mask=mask.astype(bool),
            raw_labels=raw_labels,
            confidence=confidence,
            processing_ms=elapsed,
            backend=backend_used,
            labels_present=labels_present,
        )

    def _load_backend(self) -> None:
        self._load_attempted = True
        if self.backend == "openvino":
            try:
                self._load_openvino()
                return
            except Exception as exc:
                LOGGER.warning("OpenVINO segmentation load failed, falling back to Torch CPU: %s", exc)
                self.backend = "torch"
        if self.backend == "torch":
            try:
                self._load_torch()
                return
            except Exception as exc:
                self._fallback_reason = f"Torch segmentation load failed: {exc}"
                return
        self._fallback_reason = f"Unsupported segmentation backend: {self.backend}"

    def _load_torch(self) -> None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self._model.to(torch.device("cpu"))
        self._model.eval()
        self._id2label = {int(k): normalize_label(v) for k, v in self._model.config.id2label.items()}

    def _load_openvino(self) -> None:
        from pathlib import Path

        import openvino as ov
        from transformers import AutoImageProcessor

        model_path = Path(self.openvino_model)
        xml_files = list(model_path.glob("*.xml")) if model_path.is_dir() else [model_path]
        xml_files = [p for p in xml_files if p.exists()]
        if not xml_files:
            raise FileNotFoundError(f"OpenVINO segmentation model not found: {model_path}")
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        core = ov.Core()
        self._compiled_model = core.compile_model(str(xml_files[0]), "CPU")
        self._ov_output = self._compiled_model.output(0)
        try:
            from transformers import SegformerConfig

            model_config = SegformerConfig.from_pretrained(self.model_name)
            self._id2label = {int(k): normalize_label(v) for k, v in model_config.id2label.items()}
        except Exception:
            self._id2label = {}

    def _predict_model(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self.backend == "openvino":
            return self._predict_openvino(frame)
        return self._predict_torch(frame)

    def _predict_torch(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        if self._processor is None or self._model is None:
            raise RuntimeError("Torch segmentation model is not loaded")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(torch.device("cpu")) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = functional.interpolate(
                outputs.logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(logits, dim=1)
            confidence, labels = torch.max(probs, dim=1)
        label_map = labels[0].cpu().numpy().astype(np.int32)
        conf_map = confidence[0].cpu().numpy().astype(np.float32)
        present = [self._id2label.get(int(v), str(v)) for v in np.unique(label_map)]
        return label_map, conf_map, present

    def _predict_openvino(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        from PIL import Image

        if self._processor is None or self._compiled_model is None:
            raise RuntimeError("OpenVINO segmentation model is not loaded")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="np")
        result = self._compiled_model(dict(inputs))[self._ov_output]
        logits = np.asarray(result)
        if logits.ndim != 4:
            raise RuntimeError(f"Unexpected OpenVINO segmentation output shape: {logits.shape}")
        logits_resized = np.empty((logits.shape[0], logits.shape[1], h, w), dtype=np.float32)
        for channel in range(logits.shape[1]):
            logits_resized[0, channel] = cv2.resize(logits[0, channel], (w, h), interpolation=cv2.INTER_LINEAR)
        logits_resized -= logits_resized.max(axis=1, keepdims=True)
        exp = np.exp(logits_resized)
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
        label_map = probs.argmax(axis=1)[0].astype(np.int32)
        conf_map = probs.max(axis=1)[0].astype(np.float32)
        present = [self._id2label.get(int(v), str(v)) for v in np.unique(label_map)]
        return label_map, conf_map, present

    def _classical_rgb_walkable(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = np.array(
            [
                [int(0.08 * w), h - 1],
                [int(0.92 * w), h - 1],
                [int(0.66 * w), int(0.42 * h)],
                [int(0.34 * w), int(0.42 * h)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        sample = hsv[int(0.72 * h) : h, int(0.35 * w) : int(0.65 * w)]
        if sample.size > 0:
            median = np.median(sample.reshape(-1, 3), axis=0)
            distance = np.linalg.norm((hsv - median) / np.array([20.0, 80.0, 80.0], dtype=np.float32), axis=2)
            color_mask = distance < 1.7
            mask = (mask.astype(bool) & color_mask).astype(np.uint8)
            if mask.sum() < int(0.05 * h * w):
                cv2.fillPoly(mask, [polygon], 1)
        confidence = mask.astype(np.float32) * 0.45
        return mask.astype(bool), confidence

    def _postprocess(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        cleaned = clean_binary_mask(mask, self.morphology_kernel, self.minimum_component_area)
        cleaned = keep_bottom_center_component(cleaned)
        current = cleaned.astype(np.float32)
        if self._ema is None or self._ema.shape != current.shape:
            self._ema = current
        else:
            alpha = float(np.clip(self.temporal_smoothing, 0.0, 0.95))
            self._ema = alpha * self._ema + (1.0 - alpha) * current
        return self._ema >= 0.5

