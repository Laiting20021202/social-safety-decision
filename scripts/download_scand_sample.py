#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_bev.utils import ensure_dir, read_jsonl_existing_indices, to_jsonable, write_jsonl_line


LOGGER = logging.getLogger("download_scand_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small SCAND RGB sample via HF streaming")
    parser.add_argument("--output", default="data/scand_sample", help="Output sample directory")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum saved RGB frames")
    parser.add_argument("--stride", type=int, default=10, help="Save every Nth streamed sample")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    output = Path(args.output)
    image_dir = ensure_dir(output / "images")
    manifest_path = output / "manifest.jsonl"
    existing_indices = read_jsonl_existing_indices(manifest_path)
    next_frame_id = len(sorted(image_dir.glob("frame_*.jpg"))) + 1
    saved_total = len(existing_indices)

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "mateoguaman/scand",
            split=args.split,
            streaming=True,
        )
    except Exception as exc:
        LOGGER.error("Unable to open Hugging Face dataset mateoguaman/scand: %s", exc)
        LOGGER.error("Alternative: place a local video at data/input.mp4 and run:")
        LOGGER.error("  python -m social_bev.run --input data/input.mp4 --output outputs/local_demo.mp4")
        return 2

    LOGGER.info("Writing images to %s", image_dir)
    LOGGER.info("Writing manifest to %s", manifest_path)
    try:
        for original_index, item in enumerate(tqdm(dataset, desc="streaming scand")):
            if original_index % max(1, args.stride) != 0:
                continue
            if original_index in existing_indices:
                continue
            if saved_total >= args.max_frames:
                break
            image = extract_rgb_image(item)
            if image is None:
                LOGGER.warning("No RGB image found in streamed sample index %d; skipping", original_index)
                continue
            bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            filename = f"frame_{next_frame_id:06d}.jpg"
            image_path = image_dir / filename
            if not cv2.imwrite(str(image_path), bgr):
                raise IOError(f"Failed to write {image_path}")
            width, height = image.size
            record = {
                "image_path": str(image_path.relative_to(output)),
                "original_index": int(original_index),
                "width": int(width),
                "height": int(height),
                "camera_params": pick_optional(item, ["camera_params", "camera", "intrinsics", "calibration"]),
                "trajectory_2d": pick_optional(item, ["trajectory_2d", "traj_2d", "trajectory_xy"]),
                "trajectory_3d": pick_optional(item, ["trajectory_3d", "traj_3d", "trajectory_xyz"]),
            }
            write_jsonl_line(manifest_path, to_jsonable(record))
            existing_indices.add(original_index)
            saved_total += 1
            next_frame_id += 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Existing files and manifest can be resumed with the same command.")
        return 130
    except Exception as exc:
        LOGGER.error("SCAND streaming failed: %s", exc)
        LOGGER.error("Existing partial sample is preserved. You can retry or use a local video:")
        LOGGER.error("  python -m social_bev.run --input data/input.mp4 --output outputs/local_demo.mp4")
        return 2

    LOGGER.info("Saved %d total frame records", saved_total)
    return 0


def extract_rgb_image(item: Any) -> Image.Image | None:
    direct = _extract_from_known_keys(item)
    if direct is not None:
        return direct
    return _recursive_find_image(item, depth=0)


def _extract_from_known_keys(item: Any) -> Image.Image | None:
    if not isinstance(item, dict):
        return _coerce_image(item)
    keys = [
        "image",
        "rgb",
        "rgb_image",
        "camera_image",
        "front_image",
        "front_rgb",
        "obs",
    ]
    for key in keys:
        if key in item:
            image = _coerce_image(item[key])
            if image is not None:
                return image
    return None


def _recursive_find_image(value: Any, depth: int) -> Image.Image | None:
    if depth > 4:
        return None
    image = _coerce_image(value)
    if image is not None:
        return image
    if isinstance(value, dict):
        for nested in value.values():
            image = _recursive_find_image(nested, depth + 1)
            if image is not None:
                return image
    if isinstance(value, (list, tuple)) and len(value) < 10:
        for nested in value:
            image = _recursive_find_image(nested, depth + 1)
            if image is not None:
                return image
    return None


def _coerce_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[2] in {3, 4}:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array[..., :3])
    return None


def pick_optional(item: Any, keys: list[str]) -> Any:
    if not isinstance(item, dict):
        return None
    for key in keys:
        if key in item:
            return to_jsonable(item[key])
    return None


if __name__ == "__main__":
    raise SystemExit(main())

