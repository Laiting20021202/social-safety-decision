#!/usr/bin/env python3
"""Reproducible official EdgeTAM multi-object wrapper smoke test.

This checks model load, two active object IDs, propagation, object-set rebuild,
and measured wrapper latency/VRAM. It does not measure segmentation accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from realtime_safety.edgetam_tracker.edgetam_wrapper import (  # noqa: E402
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_EDGETAM_COMMIT,
    EdgeTAMConfig,
    EdgeTAMWrapper,
)
from realtime_safety.edgetam_tracker.models import ProjectionPrompt  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=REPOSITORY_ROOT
        / "third_party"
        / "Video-Depth-Anything"
        / "assets"
        / "example_videos"
        / "Tokyo-Walk_rgb.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "edgetam_official_smoke.md",
    )
    parser.add_argument("--maximum-width", type=int, default=640)
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Bounded JPEG window length (production Koch profile uses 1)",
    )
    return parser.parse_args()


def _frames(video: Path, maximum_width: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open smoke video: {video}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < 3:
            ok, bgr = capture.read()
            if not ok:
                break
            if bgr.shape[1] > maximum_width:
                scale = maximum_width / bgr.shape[1]
                bgr = cv2.resize(
                    bgr,
                    (
                        maximum_width,
                        max(int(round(bgr.shape[0] * scale)), 1),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(np.ascontiguousarray(bgr[..., ::-1]))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Smoke video contains no readable frame: {video}")
    while len(frames) < 3:
        frames.append(frames[-1].copy())
    return frames


def _prompt(
    track_id: int,
    frame_index: int,
    shape: tuple[int, int],
    horizontal_range: tuple[float, float],
) -> ProjectionPrompt:
    height, width = shape
    left = horizontal_range[0] * width
    right = horizontal_range[1] * width
    top, bottom = 0.18 * height, 0.82 * height
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    projection_mask = np.zeros((height, width), dtype=bool)
    projection_mask[
        int(round(top)) : int(round(bottom)),
        int(round(left)) : int(round(right)),
    ] = True
    return ProjectionPrompt(
        track_id=track_id,
        frame_index=frame_index,
        box_xyxy=np.array(
            [left, top, right, bottom], dtype=np.float32
        ),
        positive_points=np.array(
            [
                [center_x, center_y],
                [center_x, 0.38 * height],
                [center_x, 0.62 * height],
            ],
            dtype=np.float32,
        ),
        projection_mask=projection_mask,
    )


def _render_report(
    *,
    video: Path,
    shape: tuple[int, int],
    torch_version: str,
    device: str,
    precision: str,
    window_size: int,
    load_ms: float,
    total_inference_ms: float,
    latencies_ms: list[float],
    peak_allocated_mib: float,
    peak_reserved_mib: float,
    cuda_extension_available: bool,
    compatibility_fallback: bool,
    summaries: list[str],
) -> str:
    lines = [
            "# Official EdgeTAM wrapper smoke",
            "",
            "This is an actual official-checkpoint execution, not a mocked",
            "predictor. It validates API/runtime integration only; the arbitrary",
            "prompts have no ground-truth masks, so accuracy remains N/A.",
            "",
            f"- Source commit: `{OFFICIAL_EDGETAM_COMMIT}`",
            f"- Checkpoint SHA-256: `{OFFICIAL_CHECKPOINT_SHA256}`",
            f"- Video: `{video}`",
            f"- Input frame shape: `{shape[1]}x{shape[0]}`",
            f"- PyTorch: `{torch_version}`",
            f"- Device / precision: `{device}` / `{precision}`",
            f"- Rolling window size: `{window_size}` frame(s)",
            f"- Model load: `{load_ms:.3f} ms`",
            (
                "- Three bounded-window inference calls: "
                f"`{total_inference_ms:.3f} ms` total "
                f"(`{3000.0 / total_inference_ms:.3f} calls/s`)"
            ),
            (
                "- Wrapper-reported inference latency: `"
                + ", ".join(f"{value:.3f}" for value in latencies_ms)
                + " ms`"
            ),
            f"- Peak CUDA allocated: `{peak_allocated_mib:.3f} MiB`",
            f"- Peak CUDA reserved: `{peak_reserved_mib:.3f} MiB`",
            (
                "- Optional upstream CUDA extension: `"
                + ("available" if cuda_extension_available else "unavailable")
                + "`"
            ),
            "",
            "## Calls",
            "",
            *[f"- {summary}" for summary in summaries],
            "",
    ]
    if compatibility_fallback:
        lines.extend(
            [
                "Pinned upstream EdgeTAM's grouped multi-object path raised a",
                "non-contiguous `view()` error with this PyTorch build. The",
                "wrapper did not patch model internals: it retried each ID in a",
                "separate official predictor state and marked each prompt mode",
                "with `+independent_state`. ROS diagnostics expose this as WARN/",
                "degraded. This compatibility mode is slower than a native",
                "grouped call.",
                "",
            ]
        )
    if not cuda_extension_available:
        lines.extend(
            [
                "The optional upstream `_C` extension was unavailable, so",
                "EdgeTAM skipped mask-hole post-processing and emitted its",
                "documented warning. Core model propagation still ran, but",
                "boundary quality may differ from a CUDA-extension build.",
                "",
            ]
        )
    lines.extend(
        [
            "The first call initializes two object IDs, the second propagates",
            "the same object set over the configured bounded window, and the third",
            "removes one ID through a safe object-set rebuild. These numbers are",
            "not ROS end-to-end FPS/latency and must not be used as a controller",
            "deadline without a synchronized RGB-D live benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    arguments = _arguments()
    frames = _frames(arguments.video, arguments.maximum_width)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; official GPU smoke not run")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wrapper = EdgeTAMWrapper(
        EdgeTAMConfig(
            repository_path=REPOSITORY_ROOT
            / "third_party"
            / "EdgeTAM",
            checkpoint_path=REPOSITORY_ROOT
            / "models"
            / "edgetam"
            / "edgetam.pt",
            model_config="configs/edgetam.yaml",
            device="cuda",
            precision="auto",
            window_size=max(int(arguments.window_size), 1),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
    )
    try:
        load_started = time.perf_counter()
        wrapper.load()
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - load_started) * 1000.0

        results = []
        inference_started = time.perf_counter()
        for frame_index, frame in enumerate(frames):
            prompt_right = _prompt(
                202, frame_index, frame.shape[:2], (0.58, 0.90)
            )
            if frame_index < 2:
                prompts = [
                    _prompt(
                        101,
                        frame_index,
                        frame.shape[:2],
                        (0.10, 0.42),
                    ),
                    prompt_right,
                ]
                active_ids = [101, 202]
            else:
                prompts = [prompt_right]
                active_ids = [202]
            result = wrapper.infer(
                frame,
                frame_index,
                frame_index / 10.0,
                prompts,
                active_track_ids=active_ids,
                timeout=120.0,
            )
            result.raise_for_error()
            if set(result.masks) != set(active_ids):
                raise RuntimeError(
                    f"Expected IDs {active_ids}, got {sorted(result.masks)}"
                )
            results.append(result)
        torch.cuda.synchronize()
        total_inference_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0
        summaries = [
            (
                f"frame {result.frame_index}: IDs={sorted(result.masks)}, "
                f"modes={result.prompt_modes}, "
                f"rebuild={result.rebuild_reason}, "
                f"window={result.window_frame_count}, "
                "mask_pixels="
                + str(
                    {
                        track_id: int(np.count_nonzero(mask))
                        for track_id, mask in result.masks.items()
                    }
                )
            )
            for result in results
        ]
        compatibility_fallback = any(
            "independent_state" in mode
            for result in results
            for mode in result.prompt_modes.values()
        )
        try:
            import sam2._C  # type: ignore[import-not-found]  # noqa: F401

            cuda_extension_available = True
        except ImportError:
            cuda_extension_available = False
        report = _render_report(
            video=arguments.video,
            shape=frames[0].shape[:2],
            torch_version=torch.__version__,
            device=wrapper.resolved_device,
            precision=wrapper.resolved_precision,
            window_size=max(int(arguments.window_size), 1),
            load_ms=load_ms,
            total_inference_ms=total_inference_ms,
            latencies_ms=[result.latency_ms for result in results],
            peak_allocated_mib=torch.cuda.max_memory_allocated()
            / (1024**2),
            peak_reserved_mib=torch.cuda.max_memory_reserved()
            / (1024**2),
            cuda_extension_available=cuda_extension_available,
            compatibility_fallback=compatibility_fallback,
            summaries=summaries,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")
        print(report)
        return 0
    finally:
        wrapper.close()


if __name__ == "__main__":
    raise SystemExit(main())
