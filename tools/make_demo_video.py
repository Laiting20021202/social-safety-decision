#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small local RGB demo video")
    parser.add_argument("--output", default="data/input.mp4", help="Output video")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise IOError(f"Failed to open {output}")
    for idx in range(args.frames):
        frame = np.full((args.height, args.width, 3), (150, 165, 175), dtype=np.uint8)
        horizon = int(args.height * 0.42)
        cv2.rectangle(frame, (0, 0), (args.width, horizon), (180, 190, 205), -1)
        road = np.array(
            [
                [int(0.08 * args.width), args.height],
                [int(0.92 * args.width), args.height],
                [int(0.62 * args.width), horizon],
                [int(0.38 * args.width), horizon],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [road], (92, 106, 96))
        cv2.line(frame, (args.width // 2, args.height), (args.width // 2, horizon), (120, 130, 118), 2)
        person_x = int(230 + idx * 1.5)
        person_y = 250
        cv2.rectangle(frame, (person_x - 16, person_y - 70), (person_x + 16, person_y), (40, 70, 200), -1)
        cv2.circle(frame, (person_x, person_y - 84), 14, (60, 80, 220), -1)
        cv2.rectangle(frame, (430, 245), (505, 315), (55, 55, 65), -1)
        cv2.rectangle(frame, (295, 280), (338, 318), (125, 80, 60), -1)
        writer.write(frame)
    writer.release()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

