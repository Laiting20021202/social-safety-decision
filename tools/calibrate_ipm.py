#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_bev.homography import compute_homography, image_to_bev_matrix, validate_quad
from social_bev.types import Calibration


CLICK_ORDER = ["far-left", "far-right", "near-right", "near-left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive four-point IPM calibration")
    parser.add_argument("--input", required=True, help="Calibration frame image")
    parser.add_argument("--output", default="configs/calibration.yaml", help="Output calibration YAML")
    parser.add_argument("--width-px", type=int, default=600)
    parser.add_argument("--height-px", type=int, default=800)
    parser.add_argument("--x-min-m", type=float, default=-3.0)
    parser.add_argument("--x-max-m", type=float, default=3.0)
    parser.add_argument("--y-min-m", type=float, default=0.0)
    parser.add_argument("--y-max-m", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError(f"Failed to read calibration frame: {args.input}")
    points: list[tuple[float, float]] = []
    window = "calibrate_ipm"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = draw_points(frame, points)
        if len(points) < 4:
            prompt = f"Click {CLICK_ORDER[len(points)]}. Keys: r reset, q quit"
        else:
            prompt = "Press s to save, r reset, q quit"
            if validate_quad(np.asarray(points, dtype=np.float32)):
                preview = make_preview(frame, points, args)
                cv2.imshow("bev preview", preview)
            else:
                prompt = "Invalid quadrilateral. Press r and click again."
        cv2.putText(canvas, prompt, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            return 1
        if key == ord("r"):
            points.clear()
            cv2.destroyWindow("bev preview")
        if key == ord("s") and len(points) == 4 and validate_quad(np.asarray(points, dtype=np.float32)):
            save_calibration(points, args)
            print(f"Saved calibration to {args.output}")
            return 0


def draw_points(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    canvas = frame.copy()
    for idx, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(canvas, CLICK_ORDER[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], isClosed=len(points) == 4, color=(0, 255, 255), thickness=2)
    return canvas


def make_world_points(args: argparse.Namespace) -> np.ndarray:
    return np.array(
        [
            [args.x_min_m, args.y_max_m],
            [args.x_max_m, args.y_max_m],
            [args.x_max_m, args.y_min_m],
            [args.x_min_m, args.y_min_m],
        ],
        dtype=np.float32,
    )


def make_bev_config(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "width_px": args.width_px,
        "height_px": args.height_px,
        "x_min_m": args.x_min_m,
        "x_max_m": args.x_max_m,
        "y_min_m": args.y_min_m,
        "y_max_m": args.y_max_m,
        "resolution_m_per_pixel": (args.x_max_m - args.x_min_m) / max(1, args.width_px),
    }


def make_preview(frame: np.ndarray, points: list[tuple[float, float]], args: argparse.Namespace) -> np.ndarray:
    image_points = np.asarray(points, dtype=np.float32)
    world_points = make_world_points(args)
    matrix = compute_homography(image_points, world_points)
    calibration = Calibration(
        homography=matrix,
        image_points=image_points,
        world_points=world_points,
        metric_bev=True,
        bev_config=make_bev_config(args),
    )
    warp = cv2.warpPerspective(frame, image_to_bev_matrix(calibration), (args.width_px, args.height_px))
    return warp


def save_calibration(points: list[tuple[float, float]], args: argparse.Namespace) -> None:
    image_points = np.asarray(points, dtype=np.float32)
    world_points = make_world_points(args)
    matrix = compute_homography(image_points, world_points)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metric_bev": True,
        "image_points": image_points.tolist(),
        "world_points": world_points.tolist(),
        "homography": matrix.tolist(),
        "bev": make_bev_config(args),
    }
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

