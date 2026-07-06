#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SegFormer semantic segmentation to OpenVINO IR")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512", help="HF model id")
    parser.add_argument("--output", default="models/segformer_b0_ade_openvino", help="Output directory")
    parser.add_argument("--input-size", type=int, default=384, help="Dummy input size")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import openvino as ov
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        processor = AutoImageProcessor.from_pretrained(args.model)
        model = SegformerForSemanticSegmentation.from_pretrained(args.model)
        model.eval()
        dummy = torch.zeros(1, 3, args.input_size, args.input_size)
        ov_model = ov.convert_model(model, example_input={"pixel_values": dummy})
        ov.save_model(ov_model, output / "segformer.xml")
        processor.save_pretrained(output)
        logging.info("Exported OpenVINO segmentation model to %s", output)
        return 0
    except Exception as exc:
        logging.error("Segmentation OpenVINO export failed: %s", exc)
        logging.error("The runtime can fall back to Torch CPU or RGB heuristic, depending on configs/default.yaml.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

