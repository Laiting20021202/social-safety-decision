from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from realtime_safety.edgetam_tracker.hand_semantic_gate import (
    HandDetection,
    HandSemanticGate,
    HandSemanticGateConfig,
    MediaPipeHandDetector,
    MediaPipeHandDetectorConfig,
    UltralyticsHandDetector,
    UltralyticsHandDetectorConfig,
    _deduplicate_hand_detections,
    _restore_rotated_detection,
    _rotate_quarter_turn,
    hand_class_ids,
    require_explicit_hand_class_ids,
)
from realtime_safety.edgetam_tracker.models import ProjectionPrompt


IMAGE_SHAPE = (100, 160)


def _prompt(
    track_id: int,
    box: tuple[int, int, int, int],
    *,
    support_box: tuple[int, int, int, int] | None = None,
) -> ProjectionPrompt:
    x1, y1, x2, y2 = support_box or box
    mask = np.zeros(IMAGE_SHAPE, dtype=bool)
    mask[y1:y2, x1:x2] = True
    positive = np.array(
        [
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            [x1 + 1, y1 + 1],
            [x2 - 2, y2 - 2],
        ],
        dtype=np.float32,
    )
    return ProjectionPrompt(
        track_id=track_id,
        frame_index=1,
        box_xyxy=np.asarray(box, dtype=np.float32),
        positive_points=positive,
        projection_mask=mask,
    )


def _hand(
    box: tuple[int, int, int, int],
    *,
    confidence: float = 0.9,
    class_name: str = "hand",
    with_mask: bool = True,
) -> HandDetection:
    mask = None
    if with_mask:
        mask = np.zeros(IMAGE_SHAPE, dtype=bool)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
    return HandDetection(
        bbox_xyxy=np.asarray(box, dtype=np.float32),
        confidence=confidence,
        class_name=class_name,
        mask=mask,
        image_size=(IMAGE_SHAPE[1], IMAGE_SHAPE[0]),
    )


def test_compact_projected_hand_is_accepted_and_selects_only_hand_pixels() -> None:
    gate = HandSemanticGate()
    prompt = _prompt(7, (35, 20, 75, 65))

    decision = gate.evaluate(
        prompt,
        [_hand((32, 18, 78, 68))],
        IMAGE_SHAPE,
    )

    assert decision.accepted
    assert decision.reason == "hand_overlap_confirmed"
    assert decision.projection_coverage == 1.0
    pixels = np.array([[40, 30], [74, 64], [90, 50]], dtype=np.float32)
    assert gate.select_supported_pixels(pixels, decision).tolist() == [
        True,
        True,
        False,
    ]


def test_giant_table_prompt_containing_small_hand_is_rejected() -> None:
    gate = HandSemanticGate()
    table_prompt = _prompt(11, (0, 0, 160, 100))

    decision = gate.evaluate(
        table_prompt,
        [_hand((20, 10, 45, 35))],
        IMAGE_SHAPE,
    )

    assert not decision.accepted
    assert decision.reason == "insufficient_hand_overlap"
    assert decision.projection_coverage < 0.05


def test_non_hand_and_low_confidence_detections_cannot_seed_edgetam() -> None:
    gate = HandSemanticGate()
    prompt = _prompt(3, (35, 20, 75, 65))

    wrong_class = gate.evaluate(
        prompt,
        [_hand((35, 20, 75, 65), class_name="mouse")],
        IMAGE_SHAPE,
    )
    weak_hand = gate.evaluate(
        prompt,
        [_hand((35, 20, 75, 65), confidence=0.2)],
        IMAGE_SHAPE,
    )

    assert not wrong_class.accepted
    assert wrong_class.reason == "no_qualified_hand_detection"
    assert not weak_hand.accepted
    assert weak_hand.reason == "no_qualified_hand_detection"


def test_successful_empty_inference_and_detector_failure_are_distinct() -> None:
    prompt = _prompt(4, (35, 20, 75, 65))
    closed_gate = HandSemanticGate()
    open_gate = HandSemanticGate(
        HandSemanticGateConfig(fail_closed_on_detector_unavailable=False)
    )

    empty = closed_gate.evaluate(prompt, [], IMAGE_SHAPE)
    unavailable = closed_gate.evaluate(prompt, None, IMAGE_SHAPE)
    fail_open = open_gate.evaluate(prompt, None, IMAGE_SHAPE)

    assert not empty.accepted
    assert empty.reason == "no_qualified_hand_detection"
    assert not unavailable.accepted
    assert unavailable.reason == "detector_unavailable"
    assert fail_open.accepted
    assert fail_open.reason == "detector_unavailable_fail_open"


def test_mask_is_required_by_default_but_bbox_only_mode_is_explicit() -> None:
    prompt = _prompt(5, (35, 20, 75, 65))
    detection = _hand((35, 20, 75, 65), with_mask=False)

    strict = HandSemanticGate().evaluate(
        prompt, [detection], IMAGE_SHAPE
    )
    bbox_only = HandSemanticGate(
        HandSemanticGateConfig(require_segmentation_mask=False)
    ).evaluate(prompt, [detection], IMAGE_SHAPE)

    assert not strict.accepted
    assert strict.reason == "no_usable_hand_mask"
    assert bbox_only.accepted


def test_filter_prompts_preserves_only_semantically_matched_track() -> None:
    gate = HandSemanticGate()
    prompts = {
        1: _prompt(1, (35, 20, 75, 65)),
        2: _prompt(2, (90, 20, 140, 70)),
    }

    accepted, decisions = gate.filter_prompts(
        prompts,
        [_hand((32, 18, 78, 68))],
        IMAGE_SHAPE,
    )

    assert set(accepted) == {1}
    assert decisions[1].accepted
    assert not decisions[2].accepted


def test_detection_mask_and_box_are_rescaled_to_rgb_resolution() -> None:
    gate = HandSemanticGate()
    prompt = _prompt(8, (40, 20, 80, 60))
    small_mask = np.zeros((50, 80), dtype=bool)
    small_mask[10:30, 20:40] = True
    detection = HandDetection(
        bbox_xyxy=np.array([20, 10, 40, 30], dtype=np.float32),
        confidence=0.95,
        mask=small_mask,
        image_size=(80, 50),
    )

    decision = gate.evaluate(prompt, [detection], IMAGE_SHAPE)

    assert decision.accepted
    assert decision.projection_coverage == 1.0


def test_custom_detector_requires_a_real_local_checkpoint(tmp_path) -> None:
    detector = UltralyticsHandDetector(
        UltralyticsHandDetectorConfig(
            model_path=str(tmp_path / "missing-hand-seg.pt")
        )
    )

    try:
        detector.load()
    except FileNotFoundError as exc:
        assert "Automatic model downloads are disabled" in str(exc)
    else:
        raise AssertionError("missing custom checkpoint must fail locally")


def test_only_explicit_hand_labels_are_resolved_from_checkpoint_names() -> None:
    names = {0: "person", 1: "mouse", 2: "hand", 3: "left_hand"}

    assert hand_class_ids(names, ("hand", "left_hand", "right_hand")) == [
        2,
        3,
    ]
    assert hand_class_ids({0: "person"}, ("hand",)) == []


def test_coco_handbag_is_rejected_as_a_hand_semantic_checkpoint() -> None:
    coco_names = {0: "person", 24: "backpack", 26: "handbag"}

    with pytest.raises(ValueError, match="not a hand-semantic model"):
        require_explicit_hand_class_ids(coco_names)


def test_hand_checkpoint_preflight_requires_an_exact_allowed_label() -> None:
    assert require_explicit_hand_class_ids(
        {0: "person", 1: "Left_Hand", 2: "glove"}
    ) == [1]


def test_mediapipe_landmarks_become_a_padded_hand_mask() -> None:
    landmarks = [
        SimpleNamespace(x=x / IMAGE_SHAPE[1], y=y / IMAGE_SHAPE[0])
        for x, y in ((45, 25), (70, 22), (78, 48), (58, 65), (38, 48))
    ]
    result = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[
            SimpleNamespace(
                classification=[SimpleNamespace(label="Left", score=0.92)]
            )
        ],
    )

    detections = MediaPipeHandDetector.detections_from_result(
        result,
        IMAGE_SHAPE,
        mask_padding_pixels=3,
    )

    assert len(detections) == 1
    assert detections[0].class_name == "left_hand"
    assert detections[0].confidence == 0.92
    assert detections[0].mask is not None
    assert detections[0].mask[45, 58]
    assert detections[0].bbox_xyxy[0] < 38
    assert detections[0].bbox_xyxy[2] > 78


def test_21_landmark_mask_keeps_finger_gaps_out_of_hand_support() -> None:
    coordinates = (
        (80, 80),
        (65, 68), (55, 58), (45, 50), (35, 45),
        (68, 55), (66, 40), (64, 25), (62, 10),
        (80, 52), (80, 35), (80, 20), (80, 5),
        (92, 55), (95, 40), (98, 25), (100, 12),
        (104, 62), (110, 50), (115, 38), (120, 28),
    )
    landmarks = [
        SimpleNamespace(x=x / IMAGE_SHAPE[1], y=y / IMAGE_SHAPE[0])
        for x, y in coordinates
    ]
    result = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[],
    )

    detection = MediaPipeHandDetector.detections_from_result(
        result,
        IMAGE_SHAPE,
        mask_padding_pixels=2,
    )[0]
    hull = cv2.convexHull(np.asarray(coordinates, dtype=np.int32))
    hull_mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    assert detection.mask is not None
    assert detection.mask[60, 80]
    assert detection.mask[10, 62]
    removed_finger_gaps = (hull_mask > 0) & ~detection.mask
    assert np.count_nonzero(removed_finger_gaps) > 500
    assert not detection.mask[21, 87]


def test_mediapipe_detector_config_rejects_unsafe_values() -> None:
    try:
        MediaPipeHandDetectorConfig(maximum_hands=0)
    except ValueError as exc:
        assert "maximum_hands" in str(exc)
    else:
        raise AssertionError("zero maximum_hands must be rejected")

    try:
        MediaPipeHandDetectorConfig(rotation_augmentation_degrees=(0, 45))
    except ValueError as exc:
        assert "rotation_augmentation_degrees" in str(exc)
    else:
        raise AssertionError("non-quarter-turn augmentation must be rejected")


def test_rotated_hand_mask_is_restored_and_duplicate_views_are_merged() -> None:
    source = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    source[24:61, 92:131] = 1
    rotated = _rotate_quarter_turn(source, 90).astype(bool)
    rows, columns = np.nonzero(rotated)
    detection = HandDetection(
        bbox_xyxy=np.array(
            [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]
        ),
        confidence=0.91,
        class_name="right_hand",
        mask=rotated,
        image_size=(rotated.shape[1], rotated.shape[0]),
    )

    restored = _restore_rotated_detection(detection, 90, IMAGE_SHAPE)
    direct = HandDetection(
        bbox_xyxy=np.array([92, 24, 131, 61]),
        confidence=0.88,
        class_name="hand",
        mask=source,
        image_size=(IMAGE_SHAPE[1], IMAGE_SHAPE[0]),
    )
    merged = _deduplicate_hand_detections([direct, restored], maximum=4)

    np.testing.assert_array_equal(restored.mask, source.astype(bool))
    np.testing.assert_array_equal(restored.bbox_xyxy, [92, 24, 131, 61])
    assert len(merged) == 1
    assert merged[0].confidence == 0.91


def test_rotation_cycle_runs_only_one_orientation_per_empty_frame() -> None:
    class EmptyHands:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, int]] = []

        def process(self, image: np.ndarray) -> SimpleNamespace:
            self.shapes.append(image.shape[:2])
            return SimpleNamespace(
                multi_hand_landmarks=[], multi_handedness=[]
            )

    backend = EmptyHands()
    detector = MediaPipeHandDetector(
        MediaPipeHandDetectorConfig(
            rotation_augmentation_degrees=(90, 0, 180),
            cycle_rotation_augmentation=True,
        )
    )
    detector._hands = backend
    image = np.zeros((100, 160, 3), dtype=np.uint8)

    assert detector.infer_rgb(image) == []
    assert detector.infer_rgb(image) == []

    assert backend.shapes == [(160, 100), (100, 160)]


def test_mediapipe_miss_is_bridged_by_optical_flow() -> None:
    landmarks = [
        SimpleNamespace(x=x / 160.0, y=y / 100.0)
        for x, y in ((48, 30), (70, 28), (78, 48), (60, 67), (42, 50))
    ]
    detected = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[
            SimpleNamespace(
                classification=[
                    SimpleNamespace(label="Right", score=0.90)
                ]
            )
        ],
    )
    empty = SimpleNamespace(
        multi_hand_landmarks=[], multi_handedness=[]
    )

    class OneDetectionThenMiss:
        def __init__(self) -> None:
            self.results = [detected, empty]

        def process(self, image: np.ndarray) -> SimpleNamespace:
            return self.results.pop(0)

    detector = MediaPipeHandDetector(
        MediaPipeHandDetectorConfig(
            mask_padding_pixels=2,
            temporal_hold_frames=3,
            minimum_flow_points=4,
        )
    )
    detector._hands = OneDetectionThenMiss()
    rng = np.random.default_rng(19)
    first_image = np.zeros((100, 160, 3), dtype=np.uint8)
    first_image[22:75, 34:86] = rng.integers(
        0, 256, size=(53, 52, 3), dtype=np.uint8
    )
    transform = np.array([[1, 0, 6], [0, 1, 4]], dtype=np.float32)
    second_image = np.stack(
        [
            cv2.warpAffine(channel, transform, (160, 100))
            for channel in np.moveaxis(first_image, -1, 0)
        ],
        axis=-1,
    )

    measured = detector.infer_rgb(first_image)
    predicted = detector.infer_rgb(second_image)

    assert len(measured) == len(predicted) == 1
    np.testing.assert_allclose(
        predicted[0].bbox_xyxy - measured[0].bbox_xyxy,
        [6, 4, 6, 4],
        atol=1.5,
    )
    assert predicted[0].confidence < measured[0].confidence
    assert not measured[0].is_prediction
    assert predicted[0].is_prediction


def test_textureless_static_hand_uses_bounded_mask_hold() -> None:
    landmarks = [
        SimpleNamespace(x=x / 160.0, y=y / 100.0)
        for x, y in ((48, 30), (70, 28), (78, 48), (60, 67), (42, 50))
    ]
    detected = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[],
    )
    empty = SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[])

    class DetectionThenMiss:
        def __init__(self) -> None:
            self.results = [detected, empty]

        def process(self, image: np.ndarray) -> SimpleNamespace:
            return self.results.pop(0)

    detector = MediaPipeHandDetector(
        MediaPipeHandDetectorConfig(temporal_hold_frames=2)
    )
    detector._hands = DetectionThenMiss()
    image = np.zeros((100, 160, 3), dtype=np.uint8)

    measured = detector.infer_rgb(image)[0]
    held = detector.infer_rgb(image)[0]

    np.testing.assert_array_equal(held.mask, measured.mask)
    np.testing.assert_array_equal(held.bbox_xyxy, measured.bbox_xyxy)
    assert held.is_prediction
    assert held.confidence < measured.confidence
