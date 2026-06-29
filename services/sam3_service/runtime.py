from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from packages.common_models import Point2D
from packages.overlay_renderer import ground_contact_point

SUPPORTED_CLASSES = ("person", "bicycle", "motorcycle", "car", "bus", "truck")
ROAD_PROMPTS = ("road", "sidewalk", "corridor", "walkable path", "traversable ground")


@dataclass
class Sam3Status:
    available: bool
    message: str
    backend: str = "unavailable"


class Sam3Runtime:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._status: Sam3Status | None = None

    def status(self) -> Sam3Status:
        if self._status is not None:
            return self._status
        self._status = self._load_status()
        return self._status

    def segment_image(
        self,
        image_path: Path,
        prompts: list[str] | None = None,
    ) -> dict[str, object]:
        status = self.status()
        if not status.available:
            raise RuntimeError(status.message)
        prompts = prompts or [*SUPPORTED_CLASSES, *ROAD_PROMPTS]
        image = Image.open(image_path).convert("RGB")
        pipeline = self._require_pipeline()
        raw = pipeline(image, text=prompts)
        return _normalize_transformers_output(raw, prompts, image.width, image.height)

    def _load_status(self) -> Sam3Status:
        try:
            import torch  # noqa: F401
            from transformers import pipeline  # noqa: F401
        except Exception as exc:
            return Sam3Status(
                available=False,
                backend="transformers",
                message=f"SAM3 unavailable: missing torch/transformers ({exc}).",
            )
        model_id = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
        hf_token = os.getenv("HF_TOKEN") or None
        try:
            self._pipeline = pipeline(
                task="mask-generation",
                model=model_id,
                token=hf_token,
                device=0 if _cuda_available() else -1,
            )
        except Exception as exc:
            return Sam3Status(
                available=False,
                backend="transformers",
                message=f"SAM3 unavailable: failed to load {model_id} ({exc}).",
            )
        return Sam3Status(available=True, backend="transformers", message=f"Loaded {model_id}.")

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            status = self.status()
            if not status.available:
                raise RuntimeError(status.message)
        return self._pipeline


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _normalize_transformers_output(
    raw: Any,
    prompts: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    masks = _as_list(raw.get("masks") if isinstance(raw, dict) else None)
    scores = _as_list(raw.get("scores") if isinstance(raw, dict) else None)
    labels = _as_list(raw.get("labels") if isinstance(raw, dict) else None)
    if not masks and isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                masks.extend(_as_list(entry.get("masks")))
                scores.extend(_as_list(entry.get("scores")))
                labels.extend(_as_list(entry.get("labels")))

    for index, mask in enumerate(masks):
        array = np.asarray(mask)
        if array.ndim > 2:
            array = array.squeeze()
        binary = array > 0
        polygon = _mask_to_bbox_polygon(binary)
        if not polygon:
            continue
        bbox = _polygon_bbox(polygon)
        label = str(labels[index]) if index < len(labels) and labels[index] is not None else ""
        class_name = _class_from_label(label, prompts)
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.5
        ground = ground_contact_point(bbox, polygon, class_name)
        items.append(
            {
                "track_id": index + 1,
                "class_name": class_name,
                "confidence": max(0.0, min(1.0, score)),
                "bounding_box": bbox,
                "mask_polygon": [point.model_dump(mode="json") for point in polygon],
                "ground_contact_point": ground.model_dump(mode="json") if ground else None,
                "source": "sam3",
                "label": label,
            }
        )
    return {
        "source": "sam3",
        "image_width": image_width,
        "image_height": image_height,
        "detections": items,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _mask_to_bbox_polygon(mask: NDArray[np.bool_]) -> list[Point2D]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return []
    x1 = float(xs.min())
    x2 = float(xs.max())
    y1 = float(ys.min())
    y2 = float(ys.max())
    return [
        Point2D(x=x1, y=y1),
        Point2D(x=x2, y=y1),
        Point2D(x=x2, y=y2),
        Point2D(x=x1, y=y2),
    ]


def _polygon_bbox(polygon: list[Point2D]) -> tuple[float, float, float, float]:
    return (
        min(point.x for point in polygon),
        min(point.y for point in polygon),
        max(point.x for point in polygon),
        max(point.y for point in polygon),
    )


def _class_from_label(label: str, prompts: list[str]) -> str:
    lowered = label.lower()
    for class_name in SUPPORTED_CLASSES:
        if class_name in lowered:
            return class_name
    for class_name in ROAD_PROMPTS:
        if class_name in lowered:
            return class_name
    for prompt in prompts:
        if prompt in SUPPORTED_CLASSES and prompt in lowered:
            return prompt
        if prompt in ROAD_PROMPTS and prompt in lowered:
            return prompt
    return "unknown"


def write_upload_to_temp(data: bytes, suffix: str = ".png") -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return Path(handle.name)
