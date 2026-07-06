#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_bev.config import load_config
from social_bev.frame_source import FrameSource
from social_bev.pipeline import SocialNavigationPipeline
from social_bev.utils import configure_logging, ensure_dir, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU social BEV pipeline")
    parser.add_argument("--input", required=True, help="Input video/images/webcam")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML")
    parser.add_argument("--calibration", default=None, help="Calibration YAML")
    parser.add_argument("--frames", type=int, default=200, help="Maximum frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--output", default="outputs/benchmark.json", help="Benchmark JSON output")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    source = FrameSource(args.input, stride=args.stride)
    pipeline = SocialNavigationPipeline(config=config, calibration_path=args.calibration)
    latencies: list[float] = []
    module_latencies: dict[str, list[float]] = {
        "segmentation": [],
        "detection": [],
        "tracking": [],
        "bev": [],
    }
    for idx, frame in enumerate(source):
        if idx >= args.frames:
            break
        result = pipeline.process_frame(frame.image, frame.timestamp)
        latencies.append(float(result.processing_ms["total"]))
        for key in module_latencies:
            module_latencies[key].append(float(result.processing_ms.get(key, 0.0)))
    if not latencies:
        raise RuntimeError(f"No frames processed from {args.input}")
    summary = {
        "frames": len(latencies),
        "average_fps": 1000.0 / statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "segmentation_latency_ms": statistics.mean(module_latencies["segmentation"]),
        "detection_latency_ms": statistics.mean(module_latencies["detection"]),
        "tracking_latency_ms": statistics.mean(module_latencies["tracking"]),
        "bev_latency_ms": statistics.mean(module_latencies["bev"]),
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(summary), indent=2))
    return 0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


if __name__ == "__main__":
    raise SystemExit(main())

