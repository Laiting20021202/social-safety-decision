from __future__ import annotations

import numpy as np
import pytest

from services.sam3_service import runtime as sam3_runtime
from services.sam3_service.runtime import (
    RuntimeModelState,
    Sam3RuntimeError,
    Sam31VideoRuntime,
    VideoSessionState,
    _mask_to_coco_rle,
    _mask_to_contour_polygon,
    _normalize_video_frame_response,
)


def test_coco_rle_preserves_true_binary_mask() -> None:
    mask = _triangle_mask()
    rle = _mask_to_coco_rle(mask)

    assert _decode_uncompressed_rle(rle) == mask.tolist()


def test_video_normalization_uses_official_object_id_and_true_mask() -> None:
    mask = _triangle_mask()
    result = _normalize_video_frame_response(
        {
            "frame_index": 4,
            "outputs": {
                "out_obj_ids": np.array([37]),
                "out_probs": np.array([0.91], dtype=np.float32),
                "out_boxes_xywh": np.array([[0.2, 0.2, 0.5, 0.6]], dtype=np.float32),
                "out_binary_masks": np.expand_dims(mask, axis=0),
            },
        },
        prompt="person",
        session=VideoSessionState(session_id="session", resource_path="/tmp/video.mp4"),
        model_state=RuntimeModelState(
            loaded=True,
            repo_id="facebook/sam3.1",
            revision="test-revision",
            checkpoint_source="test-checkpoint",
        ),
    )

    detection = result["detections"][0]
    assert detection["track_id"] == 37
    assert detection["object_id"] == 37
    assert detection["mask_rle"]
    assert detection["segmentation_type"] == "binary_mask"
    assert not _polygon_is_bbox_rectangle(detection["mask_polygon"], detection["bounding_box"])


def test_mask_contour_is_not_bbox_rectangle() -> None:
    mask = _triangle_mask()
    polygon = _mask_to_contour_polygon(mask)
    bbox = (2.0, 2.0, 10.0, 10.0)

    payload = [point.model_dump(mode="json") for point in polygon]
    assert len(payload) >= 3
    assert not _polygon_is_bbox_rectangle(payload, bbox)


def test_video_runtime_cuda_oom_unloads_and_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    video = Sam31VideoRuntime()

    def raise_oom() -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(sam3_runtime, "_cuda_available", lambda: True)
    monkeypatch.setattr(video, "_build_predictor", raise_oom)

    with pytest.raises(Sam3RuntimeError) as caught:
        video.load()

    assert caught.value.degraded
    assert not video.loaded


def _triangle_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=bool)
    for y in range(2, 10):
        mask[y, 2 : y + 1] = True
    return mask


def _decode_uncompressed_rle(rle: dict[str, object]) -> list[list[bool]]:
    height, width = rle["size"]
    values: list[int] = []
    current = 0
    for count in rle["counts"]:
        values.extend([current] * int(count))
        current = 1 - current
    array = np.array(values, dtype=bool).reshape((height, width), order="F")
    return array.tolist()


def _polygon_is_bbox_rectangle(
    polygon: object,
    bbox: object,
) -> bool:
    if not isinstance(polygon, list) or not isinstance(bbox, tuple | list) or len(bbox) != 4:
        return False
    if len(polygon) != 4:
        return False
    points = {(round(point["x"], 3), round(point["y"], 3)) for point in polygon}
    x1, y1, x2, y2 = [round(float(value), 3) for value in bbox]
    return points == {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}
