#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11 model to OpenVINO IR")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--imgsz", type=int, default=416, help="Export image size")
    parser.add_argument("--output", default="models", help="Output parent directory")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
        exported = model.export(format="openvino", imgsz=args.imgsz, device="cpu", project=args.output)
        logging.info("Exported OpenVINO YOLO model: %s", exported)
        return 0
    except Exception as exc:
        logging.error("YOLO OpenVINO export failed: %s", exc)
        logging.error("The runtime will fall back to Torch/Ultralytics CPU if configured fallback_model exists.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

