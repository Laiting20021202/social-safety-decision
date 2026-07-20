#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtime_safety.config import load_config
from realtime_safety.scheduler import RealtimePipeline
from realtime_safety.utils.gpu import gpu_info
from realtime_safety.utils.validation import validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded real-time pipeline benchmark")
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", default="realtime_fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--depth-mode",
        choices=("video_depth", "st4rtrack", "hybrid", "fast_depth", "rgbd"),
        default="fast_depth",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", default="benchmark_report.json")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    config = load_config(args.profile)
    config.device = args.device
    config.reconstruction.depth_mode = args.depth_mode
    config.video.loop = True
    validate_config(config)
    pipeline = RealtimePipeline(config)
    pipeline.start_workers()
    pipeline.start_source(int(args.source) if args.source.isdigit() else args.source)
    ram_samples: list[float] = []
    vram_samples: list[float] = []
    sample_times: list[float] = []
    stop = threading.Event()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.duration:
            ram_samples.append(psutil.Process().memory_info().rss / 2**20)
            vram_samples.append(gpu_info(args.device).allocated_mb)
            sample_times.append(time.perf_counter() - started)
            stop.wait(min(0.5, max(args.duration - (time.perf_counter() - started), 0.0)))
    finally:
        pipeline.handle_command("stop")
        final = pipeline.gui_state.read()
        errors = pipeline.errors
        pipeline.close()

    gpu = gpu_info(args.device)
    perf = final.performance
    steady_index = max(1, len(ram_samples) // 5)
    steady_ram = ram_samples[steady_index:]
    steady_vram = vram_samples[steady_index:]
    steady_times = sample_times[steady_index:]
    ram_slope = float(np.polyfit(steady_times, steady_ram, 1)[0] * 60.0) if len(steady_times) >= 2 else 0.0
    vram_slope = float(np.polyfit(steady_times, steady_vram, 1)[0] * 60.0) if len(steady_times) >= 2 else 0.0
    report = {
        "source": args.source,
        "profile_requested": args.profile,
        "profile_final": final.profile,
        "depth_mode_requested": args.depth_mode,
        "depth_source_final": final.depth_mode,
        "duration_seconds": time.perf_counter() - started,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": gpu.name,
            "gpu_total_mb": gpu.total_mb,
        },
        "performance": {
            "input_fps": perf.input_fps,
            "display_fps": perf.display_fps,
            "segmentation_fps": perf.segmentation_fps,
            "reconstruction_fps": perf.reconstruction_fps,
            "safety_fps": perf.safety_fps,
            "average_latency_ms": perf.average_latency_ms,
            "p95_latency_ms": perf.p95_latency_ms,
            "dropped_frames": perf.dropped_frames,
            "queue_size": perf.queue_size,
            "queue_capacity": perf.queue_capacity,
        },
        "memory": {
            "ram_start_mb": ram_samples[0] if ram_samples else 0.0,
            "ram_end_mb": ram_samples[-1] if ram_samples else 0.0,
            "ram_peak_mb": max(ram_samples, default=0.0),
            "vram_start_mb": vram_samples[0] if vram_samples else 0.0,
            "vram_end_mb": vram_samples[-1] if vram_samples else 0.0,
            "vram_peak_mb": max(vram_samples, default=0.0),
            "vram_after_close_mb": gpu.allocated_mb,
            "steady_ram_growth_mb": (steady_ram[-1] - steady_ram[0]) if len(steady_ram) >= 2 else 0.0,
            "steady_vram_growth_mb": (steady_vram[-1] - steady_vram[0]) if len(steady_vram) >= 2 else 0.0,
            "ram_slope_mb_per_minute": ram_slope,
            "vram_slope_mb_per_minute": vram_slope,
        },
        "runtime_fallbacks_or_errors": errors,
        "samples": len(sample_times),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
