from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


LOGGER = logging.getLogger("social_bev")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def timer_ms(name: str, timings: dict[str, float]) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = (time.perf_counter() - start) * 1000.0


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def write_jsonl_line(path: str | Path, item: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_jsonl_existing_indices(path: str | Path) -> set[int]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return set()
    indices: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "original_index" in item:
                indices.add(int(item["original_index"]))
    return indices


def letterbox(image: np.ndarray, size: tuple[int, int], color: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    """Resize an image to fit a target width/height without changing aspect ratio."""

    target_w, target_h = size
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), color, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (40, 210, 80),
    alpha: float = 0.42,
) -> np.ndarray:
    mask_bool = mask.astype(bool)
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[mask_bool] = color
    return cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)


def clean_binary_mask(mask: np.ndarray, kernel_size: int, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def keep_bottom_center_component(mask: np.ndarray) -> np.ndarray:
    """Keep the connected walkable component reachable from the lower image center."""

    binary = mask.astype(np.uint8)
    h, w = binary.shape[:2]
    if h == 0 or w == 0 or not binary.any():
        return binary.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(bool)

    bottom_y0 = max(0, int(h * 0.82))
    center_x0 = max(0, int(w * 0.35))
    center_x1 = min(w, int(w * 0.65))
    roi = labels[bottom_y0:h, center_x0:center_x1]
    label_ids, counts = np.unique(roi[roi > 0], return_counts=True)
    if len(label_ids) == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest)
    chosen = int(label_ids[int(np.argmax(counts))])
    return (labels == chosen)


def apply_cpu_thread_settings(threads: int | None) -> None:
    if threads is None or threads <= 0:
        env_value = os.environ.get("SOCIAL_BEV_CPU_THREADS")
        if env_value:
            try:
                threads = int(env_value)
            except ValueError:
                threads = None
    if threads is None or threads <= 0:
        return
    cv2.setNumThreads(int(threads))
    try:
        import torch

        torch.set_num_threads(int(threads))
    except Exception:
        LOGGER.debug("PyTorch not available while applying CPU thread settings", exc_info=True)


def image_files_in_directory(directory: str | Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in Path(directory).iterdir() if path.suffix.lower() in extensions)


def safe_cv2_imwrite(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to write image: {output_path}")


def clamp_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x, y = point
    return (float(np.clip(x, 0, max(0, width - 1))), float(np.clip(y, 0, max(0, height - 1))))

