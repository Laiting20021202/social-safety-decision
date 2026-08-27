from __future__ import annotations

"""Semantic hand gate for point-cloud generated EdgeTAM prompts.

EdgeTAM is class agnostic: it segments the object selected by a prompt, but it
cannot decide whether that object is a human hand.  This module keeps that
semantic decision separate from EdgeTAM and from ROS.  A hand detector can run
on the RGB image, while this gate verifies that its support overlaps the exact
projection of a 3D track before the prompt is admitted to EdgeTAM.

``detections=None`` deliberately means that the detector was unavailable.
``detections=[]`` means that inference succeeded and found no hand.  Keeping
those states distinct lets a safety application choose an explicit failure
policy instead of silently treating an inference failure as an empty scene.
"""

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from realtime_safety.edgetam_tracker.models import ProjectionPrompt


@dataclass(slots=True)
class HandDetection:
    """Detector output in RGB image coordinates.

    ``image_size`` follows the common ``(width, height)`` convention.  A mask
    may have a different resolution; it is resized with nearest-neighbour
    interpolation when evaluated against a prompt.
    """

    bbox_xyxy: np.ndarray
    confidence: float
    class_name: str = "hand"
    mask: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    # True when the support mask was propagated by optical flow rather than
    # produced by the hand detector in the current RGB frame.
    is_prediction: bool = False

    def __post_init__(self) -> None:
        self.bbox_xyxy = np.asarray(
            self.bbox_xyxy, dtype=np.float32
        ).reshape(4)
        if not np.isfinite(self.bbox_xyxy).all():
            raise ValueError("hand detection bbox must be finite")
        self.confidence = float(self.confidence)
        if not np.isfinite(self.confidence):
            raise ValueError("hand detection confidence must be finite")
        self.class_name = str(self.class_name).strip().lower()
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2:
                raise ValueError("hand detection mask must be 2D")
            self.mask = mask.astype(bool)
        if self.image_size is not None:
            width, height = map(int, self.image_size)
            if width <= 0 or height <= 0:
                raise ValueError("hand detection image_size must be positive")
            self.image_size = (width, height)

    @classmethod
    def from_detection(cls, detection: object) -> "HandDetection":
        """Adapt the repository's ``Detection2D`` or a compatible object."""

        return cls(
            bbox_xyxy=np.asarray(getattr(detection, "bbox_xyxy")),
            confidence=float(getattr(detection, "confidence")),
            class_name=str(getattr(detection, "class_name", "")),
            mask=getattr(detection, "mask", None),
            image_size=getattr(detection, "image_size", None),
            is_prediction=bool(getattr(detection, "is_prediction", False)),
        )


@dataclass(frozen=True, slots=True)
class HandSemanticGateConfig:
    """Acceptance thresholds for a hand-only EdgeTAM seed."""

    allowed_class_names: tuple[str, ...] = (
        "hand",
        "left_hand",
        "right_hand",
    )
    minimum_confidence: float = 0.45
    minimum_box_iou: float = 0.05
    minimum_projection_coverage: float = 0.35
    minimum_positive_point_coverage: float = 0.40
    require_segmentation_mask: bool = True
    fail_closed_on_detector_unavailable: bool = True

    def __post_init__(self) -> None:
        names = tuple(
            str(name).strip().lower()
            for name in self.allowed_class_names
            if str(name).strip()
        )
        if not names:
            raise ValueError("allowed_class_names must not be empty")
        object.__setattr__(self, "allowed_class_names", names)
        for name in (
            "minimum_confidence",
            "minimum_box_iou",
            "minimum_projection_coverage",
            "minimum_positive_point_coverage",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(slots=True)
class HandGateDecision:
    track_id: int
    accepted: bool
    reason: str
    matched_detection_index: int | None = None
    detector_confidence: float = 0.0
    box_iou: float = 0.0
    projection_coverage: float = 0.0
    positive_point_coverage: float = 0.0
    score: float = 0.0
    support_mask: np.ndarray | None = field(default=None, repr=False)


class HandSemanticGate:
    """Match hand detections to projected 3D tracks.

    A giant table prompt that merely contains a small hand box is rejected by
    the projection-coverage threshold.  Conversely, a compact 3D hand cluster
    whose projected pixels lie inside a hand mask is accepted as an EdgeTAM
    seed.  The gate never guesses a hand from colour or shape alone.
    """

    def __init__(self, config: HandSemanticGateConfig | None = None) -> None:
        self.config = config or HandSemanticGateConfig()

    def evaluate(
        self,
        prompt: ProjectionPrompt,
        detections: Sequence[HandDetection] | None,
        image_shape: tuple[int, int],
    ) -> HandGateDecision:
        height, width = _validated_image_shape(image_shape)
        if detections is None:
            accepted = not self.config.fail_closed_on_detector_unavailable
            return HandGateDecision(
                track_id=int(prompt.track_id),
                accepted=accepted,
                reason=(
                    "detector_unavailable_fail_open"
                    if accepted
                    else "detector_unavailable"
                ),
            )

        projection = _prompt_projection_mask(prompt, (height, width))
        if projection is None or not projection.any():
            return HandGateDecision(
                track_id=int(prompt.track_id),
                accepted=False,
                reason="invalid_projection_support",
            )

        prompt_box = _clipped_box(prompt.box_xyxy, width, height)
        if prompt_box is None:
            return HandGateDecision(
                track_id=int(prompt.track_id),
                accepted=False,
                reason="invalid_prompt_box",
            )

        qualifying = 0
        best: HandGateDecision | None = None
        for index, raw_detection in enumerate(detections):
            detection = (
                raw_detection
                if isinstance(raw_detection, HandDetection)
                else HandDetection.from_detection(raw_detection)
            )
            if detection.class_name not in self.config.allowed_class_names:
                continue
            if detection.confidence < self.config.minimum_confidence:
                continue
            qualifying += 1
            normalized = _detection_support(
                detection,
                (height, width),
                require_mask=self.config.require_segmentation_mask,
            )
            if normalized is None:
                continue
            detection_box, support = normalized
            intersection = int(np.count_nonzero(projection & support))
            projection_count = max(int(np.count_nonzero(projection)), 1)
            projection_coverage = intersection / projection_count
            positive_coverage = _positive_point_coverage(
                prompt.positive_points, support
            )
            box_iou = _box_iou(prompt_box, detection_box)
            accepted = (
                box_iou >= self.config.minimum_box_iou
                and projection_coverage
                >= self.config.minimum_projection_coverage
                and positive_coverage
                >= self.config.minimum_positive_point_coverage
            )
            score = float(
                detection.confidence
                * (
                    0.50 * projection_coverage
                    + 0.30 * positive_coverage
                    + 0.20 * box_iou
                )
            )
            decision = HandGateDecision(
                track_id=int(prompt.track_id),
                accepted=accepted,
                reason="hand_overlap_confirmed" if accepted else "insufficient_hand_overlap",
                matched_detection_index=index,
                detector_confidence=float(detection.confidence),
                box_iou=float(box_iou),
                projection_coverage=float(projection_coverage),
                positive_point_coverage=float(positive_coverage),
                score=score,
                support_mask=support,
            )
            if best is None or decision.score > best.score:
                best = decision

        if best is not None:
            return best
        return HandGateDecision(
            track_id=int(prompt.track_id),
            accepted=False,
            reason=(
                "no_usable_hand_mask"
                if qualifying
                else "no_qualified_hand_detection"
            ),
        )

    def filter_prompts(
        self,
        prompts: Mapping[int, ProjectionPrompt],
        detections: Sequence[HandDetection] | None,
        image_shape: tuple[int, int],
    ) -> tuple[dict[int, ProjectionPrompt], dict[int, HandGateDecision]]:
        accepted: dict[int, ProjectionPrompt] = {}
        decisions: dict[int, HandGateDecision] = {}
        for track_id, prompt in prompts.items():
            decision = self.evaluate(prompt, detections, image_shape)
            decisions[int(track_id)] = decision
            if decision.accepted:
                accepted[int(track_id)] = prompt
        return accepted, decisions

    @staticmethod
    def select_supported_pixels(
        pixels_uv: np.ndarray,
        decision: HandGateDecision,
    ) -> np.ndarray:
        """Return a boolean selector for 3D points aligned to RGB pixels."""

        pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
        if not decision.accepted:
            return np.zeros(len(pixels), dtype=bool)
        if decision.support_mask is None:
            # Explicit detector-unavailable fail-open policy.
            return np.ones(len(pixels), dtype=bool)
        height, width = decision.support_mask.shape
        rounded = np.rint(pixels).astype(np.int64)
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        selected = np.zeros(len(pixels), dtype=bool)
        valid_indices = np.flatnonzero(inside)
        selected[valid_indices] = decision.support_mask[
            rounded[valid_indices, 1], rounded[valid_indices, 0]
        ]
        return selected


@dataclass(frozen=True, slots=True)
class UltralyticsHandDetectorConfig:
    """Configuration for an opt-in, local-only hand segmentation model."""

    model_path: str
    device: str = "cuda"
    input_size: int = 512
    confidence: float = 0.35
    iou: float = 0.50
    fp16: bool = True
    maximum_detections: int = 16
    allowed_class_names: tuple[str, ...] = (
        "hand",
        "left_hand",
        "right_hand",
    )

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path must not be empty")
        if int(self.input_size) <= 0:
            raise ValueError("input_size must be positive")
        if int(self.maximum_detections) <= 0:
            raise ValueError("maximum_detections must be positive")
        for name in ("confidence", "iou"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        names = tuple(
            str(name).strip().lower()
            for name in self.allowed_class_names
            if str(name).strip()
        )
        if not names:
            raise ValueError("allowed_class_names must not be empty")
        object.__setattr__(self, "allowed_class_names", names)


class UltralyticsHandDetector:
    """Lazy custom hand-seg detector with no network-download fallback.

    This is intentionally not wired to any of the repository's COCO YOLO
    checkpoints.  ``load`` first requires a real local file and then verifies
    that the checkpoint's class table contains an explicitly allowed hand
    label.  A generic ``person`` class is never treated as a hand.
    """

    def __init__(self, config: UltralyticsHandDetectorConfig) -> None:
        self.config = config
        self.model: object | None = None
        self.device = "cpu"
        self.class_ids: list[int] = []

    def load(self) -> None:
        checkpoint = Path(self.config.model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Local hand segmentation checkpoint not found: "
                f"{checkpoint}. Automatic model downloads are disabled."
            )
        if checkpoint.stat().st_size <= 0:
            raise ValueError(f"Hand checkpoint is empty: {checkpoint}")

        import torch
        from ultralytics import YOLO

        self.device = (
            self.config.device
            if self.config.device.startswith("cuda")
            and torch.cuda.is_available()
            else "cpu"
        )
        model = YOLO(str(checkpoint), task="segment")
        names = getattr(model, "names", {})
        self.class_ids = require_explicit_hand_class_ids(
            names,
            self.config.allowed_class_names,
        )
        core = getattr(model, "model", None)
        if core is not None and hasattr(core, "fuse"):
            core.float().fuse(verbose=False)
        self.model = model

    def infer_rgb(self, rgb: np.ndarray) -> list[HandDetection]:
        if self.model is None:
            raise RuntimeError("Call load() before hand inference")
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("hand detector input must be an RGB HxWx3 image")
        image = np.ascontiguousarray(image[..., :3])
        height, width = image.shape[:2]

        import torch

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.startswith("cuda") and self.config.fp16
            else contextlib.nullcontext()
        )
        # Ultralytics' ndarray input follows OpenCV's BGR convention.
        bgr = np.ascontiguousarray(image[..., ::-1])
        with torch.inference_mode(), autocast:
            results = self.model.predict(
                source=bgr,
                imgsz=int(self.config.input_size),
                conf=float(self.config.confidence),
                iou=float(self.config.iou),
                device=self.device,
                classes=self.class_ids,
                retina_masks=True,
                verbose=False,
                max_det=int(self.config.maximum_detections),
            )
        if not results:
            return []
        result = results[0]
        boxes_result = getattr(result, "boxes", None)
        if boxes_result is None or len(boxes_result) == 0:
            return []
        boxes = boxes_result.xyxy.detach().float().cpu().numpy()
        classes = boxes_result.cls.detach().int().cpu().numpy()
        confidences = boxes_result.conf.detach().float().cpu().numpy()
        masks_result = getattr(result, "masks", None)
        masks = (
            None
            if masks_result is None
            else masks_result.data.detach().float().cpu().numpy()
        )
        names = dict(getattr(self.model, "names", {}))
        detections: list[HandDetection] = []
        for index, (box, class_id, confidence) in enumerate(
            zip(boxes, classes, confidences)
        ):
            name = str(names.get(int(class_id), "")).strip().lower()
            if name not in self.config.allowed_class_names:
                continue
            mask = None
            if masks is not None and index < len(masks):
                mask = cv2.resize(
                    masks[index],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
            detections.append(
                HandDetection(
                    bbox_xyxy=box,
                    confidence=float(confidence),
                    class_name=name,
                    mask=mask,
                    image_size=(width, height),
                )
            )
        return detections

    def close(self) -> None:
        self.model = None
        self.class_ids = []
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass


@dataclass(frozen=True, slots=True)
class MediaPipeHandDetectorConfig:
    """Configuration for the shared RGB hand semantic gate.

    MediaPipe supplies 21 hand landmarks rather than an instance mask.  The
    detector below converts the palm and finger bones into a padded hand
    support mask; 3D depth/foreground gates still own the final geometry.
    """

    maximum_hands: int = 4
    minimum_detection_confidence: float = 0.50
    minimum_tracking_confidence: float = 0.45
    model_complexity: int = 0
    mask_padding_pixels: int = 4
    static_image_mode: bool = False
    rotation_augmentation_degrees: tuple[int, ...] = (0,)
    cycle_rotation_augmentation: bool = False
    # MediaPipe's landmark detector can miss a blurred hand for one or more
    # frames. Track the last accepted support mask with sparse optical flow so
    # both selectable pipelines keep a responsive box during that short gap.
    temporal_hold_frames: int = 8
    temporal_confidence_decay: float = 0.94
    minimum_flow_points: int = 4
    maximum_flow_error: float = 30.0

    def __post_init__(self) -> None:
        if int(self.maximum_hands) < 1:
            raise ValueError("maximum_hands must be at least one")
        if int(self.model_complexity) not in {0, 1}:
            raise ValueError("model_complexity must be 0 or 1")
        if int(self.mask_padding_pixels) < 0:
            raise ValueError("mask_padding_pixels cannot be negative")
        if int(self.temporal_hold_frames) < 0:
            raise ValueError("temporal_hold_frames cannot be negative")
        if int(self.minimum_flow_points) < 1:
            raise ValueError("minimum_flow_points must be at least one")
        if float(self.maximum_flow_error) <= 0.0:
            raise ValueError("maximum_flow_error must be positive")
        if not 0.0 < float(self.temporal_confidence_decay) <= 1.0:
            raise ValueError(
                "temporal_confidence_decay must be within (0, 1]"
            )
        rotations = tuple(int(value) for value in self.rotation_augmentation_degrees)
        if not rotations or any(value not in {0, 90, -90, 180} for value in rotations):
            raise ValueError(
                "rotation_augmentation_degrees must contain only 0, 90, -90, or 180"
            )
        for name in (
            "minimum_detection_confidence",
            "minimum_tracking_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


class MediaPipeHandDetector:
    """Persistent MediaPipe hand landmarker exposed as hand instance masks."""

    def __init__(
        self, config: MediaPipeHandDetectorConfig | None = None
    ) -> None:
        self.config = config or MediaPipeHandDetectorConfig()
        self._hands: object | None = None
        self._lock = threading.Lock()
        self._rotation_index = 0
        self._previous_gray: np.ndarray | None = None
        self._temporal_detections: list[HandDetection] = []
        self._temporal_misses = 0

    @property
    def loaded(self) -> bool:
        return self._hands is not None

    def load(self) -> None:
        if self._hands is not None:
            return
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=bool(self.config.static_image_mode),
            max_num_hands=int(self.config.maximum_hands),
            model_complexity=int(self.config.model_complexity),
            min_detection_confidence=float(
                self.config.minimum_detection_confidence
            ),
            min_tracking_confidence=float(
                self.config.minimum_tracking_confidence
            ),
        )

    def infer_rgb(self, rgb: np.ndarray) -> list[HandDetection]:
        if self._hands is None:
            raise RuntimeError("Call load() before MediaPipe hand inference")
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("hand detector input must be an RGB HxWx3 image")
        image = np.ascontiguousarray(image[..., :3])
        detections: list[HandDetection] = []
        rotations = tuple(
            dict.fromkeys(
                int(value)
                for value in self.config.rotation_augmentation_degrees
            )
        )
        selected_rotations = rotations
        if self.config.cycle_rotation_augmentation and len(rotations) > 1:
            selected_rotations = (
                rotations[self._rotation_index % len(rotations)],
            )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        with self._lock:
            for degrees in selected_rotations:
                rotated = _rotate_quarter_turn(image, degrees)
                results = self._hands.process(rotated)
                rotated_detections = self.detections_from_result(
                    results,
                    rotated.shape[:2],
                    mask_padding_pixels=int(self.config.mask_padding_pixels),
                )
                detections.extend(
                    _restore_rotated_detection(
                        detection,
                        degrees,
                        image.shape[:2],
                    )
                    for detection in rotated_detections
                )
            accepted = _deduplicate_hand_detections(
                detections,
                maximum=int(self.config.maximum_hands),
            )
            if accepted:
                self._previous_gray = gray
                self._temporal_detections = [
                    _copy_hand_detection(item) for item in accepted
                ]
                self._temporal_misses = 0
                return accepted

            if (
                self.config.cycle_rotation_augmentation
                and len(rotations) > 1
            ):
                # Continue searching the neural detector's quarter-turn views
                # even if optical flow can bridge the current missed frame.
                self._rotation_index = (
                    self._rotation_index + 1
                ) % len(rotations)

            tracked = self._track_temporal(gray)
            self._previous_gray = gray
            if tracked:
                self._temporal_detections = [
                    _copy_hand_detection(item) for item in tracked
                ]
                self._temporal_misses += 1
                return tracked
            self._temporal_detections.clear()
            self._temporal_misses = 0
            return []

    def _track_temporal(
        self, current_gray: np.ndarray
    ) -> list[HandDetection]:
        previous_gray = self._previous_gray
        if (
            previous_gray is None
            or not self._temporal_detections
            or self.config.temporal_hold_frames <= 0
            or self._temporal_misses >= self.config.temporal_hold_frames
            or previous_gray.shape != current_gray.shape
        ):
            return []

        tracked: list[HandDetection] = []
        for detection in self._temporal_detections:
            candidate = _flow_track_hand_detection(
                detection,
                previous_gray,
                current_gray,
                confidence_decay=float(
                    self.config.temporal_confidence_decay
                ),
                minimum_points=int(self.config.minimum_flow_points),
                maximum_error=float(self.config.maximum_flow_error),
            )
            if candidate is None:
                # A texture-poor simulated hand can provide too few optical
                # flow corners even though it has not moved.  Hold the most
                # recent semantic mask for the same bounded miss window rather
                # than alternating the obstacle on/off every detector frame.
                candidate = _copy_hand_detection(detection)
                candidate.confidence *= float(
                    self.config.temporal_confidence_decay
                )
                candidate.is_prediction = True
            if candidate is not None:
                tracked.append(candidate)
        return _deduplicate_hand_detections(
            tracked,
            maximum=int(self.config.maximum_hands),
        )

    @staticmethod
    def detections_from_result(
        result: object,
        image_shape: tuple[int, int],
        *,
        mask_padding_pixels: int = 4,
    ) -> list[HandDetection]:
        """Convert MediaPipe output; public for deterministic unit tests."""

        height, width = _validated_image_shape(image_shape)
        landmark_sets = list(
            getattr(result, "multi_hand_landmarks", None) or ()
        )
        handedness_sets = list(
            getattr(result, "multi_handedness", None) or ()
        )
        detections: list[HandDetection] = []
        for index, landmark_set in enumerate(landmark_sets):
            landmarks = list(getattr(landmark_set, "landmark", ()) or ())
            pixels = np.asarray(
                [
                    (
                        float(getattr(landmark, "x", np.nan)) * width,
                        float(getattr(landmark, "y", np.nan)) * height,
                    )
                    for landmark in landmarks
                ],
                dtype=np.float32,
            ).reshape(-1, 2)
            finite = np.isfinite(pixels).all(axis=1)
            pixels = pixels[finite]
            if len(pixels) < 3:
                continue
            pixels[:, 0] = np.clip(pixels[:, 0], 0.0, width - 1.0)
            pixels[:, 1] = np.clip(pixels[:, 1], 0.0, height - 1.0)
            mask = np.zeros((height, width), dtype=np.uint8)
            integer_pixels = np.rint(pixels).astype(np.int32)
            if len(integer_pixels) == 21:
                # A convex hull fills every gap between fingers and therefore
                # labels the visible table as a hand.  Fill only the palm and
                # draw the actual MediaPipe kinematic chains.
                palm = integer_pixels[[0, 1, 2, 5, 9, 13, 17]]
                cv2.fillConvexPoly(mask, cv2.convexHull(palm), 1)
                palm_width = float(
                    np.linalg.norm(pixels[5] - pixels[17])
                )
                bone_width = int(
                    np.clip(round(0.18 * palm_width), 3, 18)
                )
                connections = (
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (17, 0),
                )
                for start, end in connections:
                    cv2.line(
                        mask,
                        tuple(integer_pixels[start]),
                        tuple(integer_pixels[end]),
                        1,
                        bone_width,
                        lineType=cv2.LINE_8,
                    )
                radius = max(2, bone_width // 2)
                for point in integer_pixels:
                    cv2.circle(mask, tuple(point), radius, 1, -1)
            else:
                # Deterministic tests and older adapters may provide only a
                # contour instead of the canonical 21 landmarks.
                hull = cv2.convexHull(integer_pixels)
                cv2.fillConvexPoly(mask, hull, 1)
            padding = max(int(mask_padding_pixels), 0)
            if padding:
                kernel_size = padding * 2 + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
            rows, columns = np.nonzero(mask)
            if len(columns) == 0:
                continue
            confidence = 1.0
            class_name = "hand"
            if index < len(handedness_sets):
                classes = list(
                    getattr(handedness_sets[index], "classification", ())
                    or ()
                )
                if classes:
                    confidence = float(
                        np.clip(
                            getattr(classes[0], "score", 1.0),
                            0.0,
                            1.0,
                        )
                    )
                    label = str(
                        getattr(classes[0], "label", "")
                    ).strip().lower()
                    if label in {"left", "right"}:
                        class_name = f"{label}_hand"
            detections.append(
                HandDetection(
                    bbox_xyxy=np.asarray(
                        (
                            float(columns.min()),
                            float(rows.min()),
                            float(columns.max() + 1),
                            float(rows.max() + 1),
                        ),
                        dtype=np.float32,
                    ),
                    confidence=confidence,
                    class_name=class_name,
                    mask=mask.astype(bool),
                    image_size=(width, height),
                )
            )
        return detections

    def close(self) -> None:
        hands = self._hands
        self._hands = None
        self._previous_gray = None
        self._temporal_detections.clear()
        self._temporal_misses = 0
        close = getattr(hands, "close", None)
        if callable(close):
            close()


def _copy_hand_detection(detection: HandDetection) -> HandDetection:
    return HandDetection(
        bbox_xyxy=detection.bbox_xyxy.copy(),
        confidence=float(detection.confidence),
        class_name=detection.class_name,
        mask=(
            None
            if detection.mask is None
            else np.asarray(detection.mask, dtype=bool).copy()
        ),
        image_size=detection.image_size,
        is_prediction=bool(detection.is_prediction),
    )


def _flow_track_hand_detection(
    detection: HandDetection,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    confidence_decay: float,
    minimum_points: int,
    maximum_error: float,
) -> HandDetection | None:
    """Translate a recent hand mask with robust sparse optical flow.

    This is intentionally a bounded gap bridge, not a semantic detector. The
    caller discards it after ``temporal_hold_frames`` and confidence decays on
    every predicted frame. If a textureless hand yields no usable feature,
    zero displacement retains the conservative last support briefly instead
    of making a confirmed obstacle blink out immediately.
    """

    mask = detection.mask
    if mask is None or mask.shape != previous_gray.shape or not mask.any():
        return None
    if previous_gray.shape != current_gray.shape:
        return None

    feature_mask = np.asarray(mask, dtype=np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=64,
        qualityLevel=0.01,
        minDistance=3.0,
        mask=feature_mask,
        blockSize=5,
    )
    displacement = np.zeros(2, dtype=np.float32)
    if corners is not None and len(corners) >= int(minimum_points):
        try:
            moved, status, errors = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                corners,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.01,
                ),
            )
        except cv2.error:
            moved = status = errors = None
        if moved is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            if errors is not None:
                valid &= np.isfinite(errors.reshape(-1))
                valid &= errors.reshape(-1) <= float(maximum_error)
            previous_points = corners.reshape(-1, 2)
            moved_points = moved.reshape(-1, 2)
            valid &= np.isfinite(previous_points).all(axis=1)
            valid &= np.isfinite(moved_points).all(axis=1)
            deltas = moved_points[valid] - previous_points[valid]
            if len(deltas) >= int(minimum_points):
                median = np.median(deltas, axis=0)
                deviations = np.linalg.norm(deltas - median, axis=1)
                mad = float(np.median(deviations))
                inliers = deviations <= max(2.5 * mad, 1.5)
                if int(np.count_nonzero(inliers)) >= int(minimum_points):
                    displacement = np.median(
                        deltas[inliers], axis=0
                    ).astype(np.float32)

    height, width = previous_gray.shape
    if (
        not np.isfinite(displacement).all()
        or abs(float(displacement[0])) > width * 0.5
        or abs(float(displacement[1])) > height * 0.5
    ):
        return None
    transform = np.asarray(
        (
            (1.0, 0.0, float(displacement[0])),
            (0.0, 1.0, float(displacement[1])),
        ),
        dtype=np.float32,
    )
    shifted = cv2.warpAffine(
        feature_mask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    rows, columns = np.nonzero(shifted)
    if not len(columns):
        return None
    return HandDetection(
        bbox_xyxy=np.asarray(
            (
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            ),
            dtype=np.float32,
        ),
        confidence=float(
            np.clip(
                detection.confidence * float(confidence_decay),
                0.0,
                1.0,
            )
        ),
        class_name=detection.class_name,
        mask=shifted,
        image_size=(width, height),
        is_prediction=True,
    )


def _rotate_quarter_turn(image: np.ndarray, degrees: int) -> np.ndarray:
    value = int(degrees)
    if value == 0:
        return image
    if value == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if value == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if value == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"unsupported quarter-turn rotation: {degrees}")


def _restore_rotated_detection(
    detection: HandDetection,
    degrees: int,
    original_shape: tuple[int, int],
) -> HandDetection:
    """Map a rotated MediaPipe support mask back to the source RGB frame."""

    height, width = _validated_image_shape(original_shape)
    if detection.mask is None:
        raise ValueError("rotated MediaPipe detection must include a mask")
    inverse = {0: 0, 90: -90, -90: 90, 180: 180}[int(degrees)]
    mask = _rotate_quarter_turn(
        detection.mask.astype(np.uint8), inverse
    ).astype(bool)
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return HandDetection(
            bbox_xyxy=np.zeros(4, dtype=np.float32),
            confidence=detection.confidence,
            class_name=detection.class_name,
            mask=mask,
            image_size=(width, height),
            is_prediction=bool(detection.is_prediction),
        )
    return HandDetection(
        bbox_xyxy=np.asarray(
            (
                columns.min(),
                rows.min(),
                columns.max() + 1,
                rows.max() + 1,
            ),
            dtype=np.float32,
        ),
        confidence=detection.confidence,
        class_name=detection.class_name,
        mask=mask,
        image_size=(width, height),
        is_prediction=bool(detection.is_prediction),
    )


def _deduplicate_hand_detections(
    detections: Sequence[HandDetection],
    *,
    maximum: int,
) -> list[HandDetection]:
    accepted: list[HandDetection] = []
    for detection in sorted(
        detections, key=lambda item: item.confidence, reverse=True
    ):
        if detection.mask is None or not detection.mask.any():
            continue
        duplicate = False
        for previous in accepted:
            assert previous.mask is not None
            intersection = int(np.count_nonzero(detection.mask & previous.mask))
            union = int(np.count_nonzero(detection.mask | previous.mask))
            if union and intersection / union >= 0.45:
                duplicate = True
                break
        if not duplicate:
            accepted.append(detection)
        if len(accepted) >= max(int(maximum), 1):
            break
    return accepted


def hand_class_ids(
    names: Mapping[int, str] | Sequence[str],
    allowed_class_names: Iterable[str],
) -> list[int]:
    """Resolve only explicit hand labels from a model class table."""

    allowed = {
        str(name).strip().lower()
        for name in allowed_class_names
        if str(name).strip()
    }
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    return [
        int(class_id)
        for class_id, class_name in items
        if str(class_name).strip().lower() in allowed
    ]


def require_explicit_hand_class_ids(
    names: Mapping[int, str] | Sequence[str],
    allowed_class_names: Iterable[str] = (
        "hand",
        "left_hand",
        "right_hand",
    ),
) -> list[int]:
    """Validate that a YOLO checkpoint has an explicit hand class.

    Generic COCO segmentation weights contain ``handbag`` but no human-hand
    class.  Substring matching would therefore be unsafe.  This helper keeps
    the exact-label rule in one place for every local hand-checkpoint preflight.
    """

    allowed = tuple(
        str(name).strip().lower()
        for name in allowed_class_names
        if str(name).strip()
    )
    class_ids = hand_class_ids(names, allowed)
    if class_ids:
        return class_ids
    available_names = names.values() if isinstance(names, Mapping) else names
    available = ", ".join(sorted({str(value) for value in available_names}))
    raise ValueError(
        "Checkpoint is not a hand-semantic model: no explicit hand class "
        f"{allowed!r}; available classes: {available or '<none>'}"
    )


def _validated_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = map(int, image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    return height, width


def _prompt_projection_mask(
    prompt: ProjectionPrompt,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    mask = prompt.projection_mask
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_shape:
        return None
    return mask


def _detection_support(
    detection: HandDetection,
    image_shape: tuple[int, int],
    *,
    require_mask: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = image_shape
    box = detection.bbox_xyxy.astype(np.float32).copy()
    if detection.image_size is not None:
        source_width, source_height = detection.image_size
        box[[0, 2]] *= width / float(source_width)
        box[[1, 3]] *= height / float(source_height)
    clipped = _clipped_box(box, width, height)
    if clipped is None:
        return None

    box_mask = np.zeros((height, width), dtype=bool)
    x1 = max(int(np.floor(clipped[0])), 0)
    y1 = max(int(np.floor(clipped[1])), 0)
    x2 = min(int(np.ceil(clipped[2])), width)
    y2 = min(int(np.ceil(clipped[3])), height)
    if x2 <= x1 or y2 <= y1:
        return None
    box_mask[y1:y2, x1:x2] = True

    if detection.mask is None:
        if require_mask:
            return None
        return clipped, box_mask
    mask = detection.mask.astype(np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    support = mask.astype(bool) & box_mask
    if not support.any():
        return None
    return clipped, support


def _clipped_box(
    box_xyxy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4).copy()
    if not np.isfinite(box).all():
        return None
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    first_area = max(float(first[2] - first[0]), 0.0) * max(
        float(first[3] - first[1]), 0.0
    )
    second_area = max(float(second[2] - second[0]), 0.0) * max(
        float(second[3] - second[1]), 0.0
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _positive_point_coverage(
    positive_points: np.ndarray,
    support: np.ndarray,
) -> float:
    points = np.asarray(positive_points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return 0.0
    height, width = support.shape
    rounded = np.rint(points).astype(np.int64)
    inside = (
        np.isfinite(points).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    supported = np.zeros(len(points), dtype=bool)
    valid_indices = np.flatnonzero(inside)
    supported[valid_indices] = support[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ]
    return float(np.mean(supported))


__all__ = [
    "HandDetection",
    "HandGateDecision",
    "HandSemanticGate",
    "HandSemanticGateConfig",
    "MediaPipeHandDetector",
    "MediaPipeHandDetectorConfig",
    "UltralyticsHandDetector",
    "UltralyticsHandDetectorConfig",
    "hand_class_ids",
    "require_explicit_hand_class_ids",
]
