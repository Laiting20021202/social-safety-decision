from __future__ import annotations

import numpy as np

from realtime_safety.config import SegmentationConfig
from realtime_safety.edgetam_tracker.hand_semantic_gate import HandDetection
from realtime_safety.pipeline.segmentation import UltralyticsSegmentationBackend
from realtime_safety.types import FramePacket


def _frame() -> FramePacket:
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    return FramePacket(
        frame_index=1,
        source_timestamp=12.5,
        capture_timestamp=12.5,
        bgr=rgb[..., ::-1].copy(),
        rgb=rgb,
        original_fps=20.0,
        original_width=80,
        original_height=60,
    )


def test_hand_only_yolo_path_never_returns_generic_coco_obstacles() -> None:
    backend = UltralyticsSegmentationBackend(
        SegmentationConfig(hand_only=True),
        device="cpu",
    )
    hand_mask = np.zeros((60, 80), dtype=bool)
    hand_mask[10:35, 20:50] = True

    class FakeHandDetector:
        def infer_rgb(self, _rgb: np.ndarray) -> list[HandDetection]:
            return [
                HandDetection(
                    bbox_xyxy=np.array((20, 10, 50, 35), dtype=np.float32),
                    confidence=0.91,
                    class_name="right_hand",
                    mask=hand_mask,
                    image_size=(80, 60),
                )
            ]

    backend.hand_detector = FakeHandDetector()  # type: ignore[assignment]
    backend._track_classes = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("generic COCO YOLO must not run in hand-only mode")
    )

    detections = backend.track_obstacles(_frame())

    assert len(detections) == 1
    assert detections[0].class_name == "hand"
    assert detections[0].class_id == -1
    assert detections[0].confidence == 0.91
    assert np.array_equal(detections[0].mask, hand_mask)


def test_hand_only_yolo_path_outputs_nothing_without_hand_confirmation() -> None:
    backend = UltralyticsSegmentationBackend(
        SegmentationConfig(hand_only=True),
        device="cpu",
    )

    class EmptyHandDetector:
        def infer_rgb(self, _rgb: np.ndarray) -> list[HandDetection]:
            return []

    backend.hand_detector = EmptyHandDetector()  # type: ignore[assignment]
    backend._track_classes = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("generic COCO YOLO must not run in hand-only mode")
    )

    assert backend.track_obstacles(_frame()) == []


def test_hand_only_yolo_marks_optical_flow_support_as_prediction() -> None:
    backend = UltralyticsSegmentationBackend(
        SegmentationConfig(hand_only=True),
        device="cpu",
    )
    hand_mask = np.zeros((60, 80), dtype=bool)
    hand_mask[12:30, 62:80] = True

    class PredictedHandDetector:
        def infer_rgb(self, _rgb: np.ndarray) -> list[HandDetection]:
            return [
                HandDetection(
                    bbox_xyxy=np.array((62, 12, 80, 30), dtype=np.float32),
                    confidence=0.72,
                    mask=hand_mask,
                    image_size=(80, 60),
                    is_prediction=True,
                )
            ]

    backend.hand_detector = PredictedHandDetector()  # type: ignore[assignment]

    detections = backend.track_obstacles(_frame())

    assert len(detections) == 1
    assert detections[0].is_prediction
