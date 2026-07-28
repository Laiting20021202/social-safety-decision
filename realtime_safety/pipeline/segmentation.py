from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from realtime_safety.config import SegmentationConfig
from realtime_safety.types import Detection2D, FramePacket
from realtime_safety.utils.timing import CudaEventTimer

LOGGER = logging.getLogger(__name__)


class SegmentationBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def infer(self, frame: FramePacket) -> list[Detection2D]: ...

    def infer_people(self, frame: FramePacket) -> list[Detection2D]:
        return [detection for detection in self.infer(frame) if detection.class_name == "person"]

    def track_people(self, frame: FramePacket) -> list[Detection2D]:
        """Track people in consecutive frames.

        Backends without a native tracker retain the previous person-only
        inference behaviour; the scheduler will assign IDs locally.
        """

        return self.infer_people(frame)

    def track_obstacles(self, frame: FramePacket) -> list[Detection2D]:
        """Track every model class accepted as a collision obstacle."""

        return self.infer(frame)

    def reset_tracking(self) -> None:
        return None

    def close(self) -> None:
        return None


class UltralyticsSegmentationBackend(SegmentationBackend):
    """Persistent YOLO instance-segmentation backend with CUDA mixed precision."""

    OBSTACLE_CLASSES = {
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck",
        "chair",
        "couch",
        "backpack",
        "handbag",
        "suitcase",
        "bench",
        "dog",
        "cat",
    }
    CLASS_ALIASES = {
        "motorcycle": "vehicle",
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "couch": "chair",
        "backpack": "bag",
        "handbag": "bag",
    }

    def __init__(self, config: SegmentationConfig, device: str = "cuda") -> None:
        self.config = config
        self.requested_device = device
        self.device = "cpu"
        self.model = None
        self.names: dict[int, str] = {}
        self.last_gpu_ms = 0.0

    def load(self) -> None:
        import torch
        from ultralytics import YOLO

        self.device = self.requested_device if self.requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = YOLO(self.config.model, task="segment")
        # Fuse in FP32 before Ultralytics builds its FP16 AutoBackend. This avoids
        # a Conv/BN dtype mismatch on PyTorch 2.9 during the first prediction.
        if hasattr(self.model, "model") and hasattr(self.model.model, "fuse"):
            self.model.model.float().fuse(verbose=False)
        self.names = dict(self.model.names)
        LOGGER.info("Loaded %s on %s", self.config.model, self.device)

    def warmup(self) -> None:
        if self.model is None:
            self.load()
        dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
        self._predict(dummy, verbose=False)

    def _predict(self, bgr: np.ndarray, verbose: bool = False, classes: list[int] | None = None):
        if self.model is None:
            raise RuntimeError("Segmentation model has not been loaded")
        import torch

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.startswith("cuda") and self.config.fp16
            else contextlib.nullcontext()
        )
        timer = CudaEventTimer(self.device.startswith("cuda"))
        with torch.inference_mode(), autocast, timer:
            results = self.model.predict(
                source=bgr,
                imgsz=self.config.input_size,
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.device,
                quantize=16 if self.device.startswith("cuda") and self.config.fp16 else None,
                retina_masks=False,
                verbose=verbose,
                max_det=100,
                classes=classes,
            )
        self.last_gpu_ms = timer.elapsed_ms
        return results

    def infer(self, frame: FramePacket) -> list[Detection2D]:
        return self._results_to_detections(self._predict(frame.bgr), frame)

    def infer_people(self, frame: FramePacket) -> list[Detection2D]:
        person_ids = [class_id for class_id, name in self.names.items() if name == "person"]
        return self._results_to_detections(self._predict(frame.bgr, classes=person_ids or None), frame)

    def track_people(self, frame: FramePacket) -> list[Detection2D]:
        """Use ByteTrack's low-confidence recovery while only tracking people."""

        return self._track_classes(frame, {"person"})

    def track_obstacles(self, frame: FramePacket) -> list[Detection2D]:
        """Track all accepted obstacle classes with one persistent ByteTrack state."""

        return self._track_classes(frame, self.OBSTACLE_CLASSES)

    def _track_classes(
        self,
        frame: FramePacket,
        class_names: set[str],
    ) -> list[Detection2D]:
        if self.model is None:
            raise RuntimeError("Segmentation model has not been loaded")
        import torch

        class_ids = [
            class_id
            for class_id, name in self.names.items()
            if name in class_names
        ]
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.startswith("cuda") and self.config.fp16
            else contextlib.nullcontext()
        )
        timer = CudaEventTimer(self.device.startswith("cuda"))
        with torch.inference_mode(), autocast, timer:
            results = self.model.track(
                source=frame.bgr,
                imgsz=self.config.input_size,
                conf=self.config.tracking_confidence,
                iou=self.config.iou,
                device=self.device,
                quantize=16 if self.device.startswith("cuda") and self.config.fp16 else None,
                classes=class_ids or None,
                persist=True,
                tracker=self.config.tracker_config or "bytetrack.yaml",
                # Full-resolution masks materially reduce green-arm pixels
                # leaking into nearby person/obstacle clouds.
                retina_masks=True,
                verbose=False,
                max_det=100,
            )
        self.last_gpu_ms = timer.elapsed_ms
        return self._results_to_detections(results, frame)

    def _results_to_detections(self, results, frame: FramePacket) -> list[Detection2D]:
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().float().cpu().numpy()
        classes = result.boxes.cls.detach().int().cpu().numpy()
        confidences = result.boxes.conf.detach().float().cpu().numpy()
        track_ids = (
            result.boxes.id.detach().int().cpu().numpy()
            if result.boxes.id is not None
            else None
        )
        masks = None
        if result.masks is not None:
            masks = result.masks.data.detach().float().cpu().numpy()
        detections: list[Detection2D] = []
        for index, (bbox, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
            original_name = self.names.get(int(class_id), "unknown_obstacle")
            if original_name not in self.OBSTACLE_CLASSES:
                continue
            class_name = self.CLASS_ALIASES.get(original_name, original_name)
            mask = None
            if masks is not None and index < len(masks):
                mask = cv2.resize(
                    masks[index],
                    (frame.original_width, frame.original_height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
            centroid = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
            detections.append(
                Detection2D(
                    bbox_xyxy=bbox.astype(np.float32),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    centroid_xy=centroid,
                    timestamp=frame.source_timestamp,
                    mask=mask,
                    track_id=int(track_ids[index]) if track_ids is not None else None,
                    image_size=(frame.original_width, frame.original_height),
                )
            )
        return detections

    def reset_tracking(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_segmentation_backend(config: SegmentationConfig, device: str) -> SegmentationBackend:
    return UltralyticsSegmentationBackend(config, device=device)
