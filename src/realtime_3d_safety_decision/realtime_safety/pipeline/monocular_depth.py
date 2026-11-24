from __future__ import annotations

import contextlib
import logging
import time

import cv2
import numpy as np
from PIL import Image

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.pointcloud import (
    depth_to_pointmap,
    relative_inverse_depth,
    resize_for_pointmap,
    voxel_downsample,
)
from realtime_safety.types import FramePacket, PointCloudFrame
from realtime_safety.utils.timing import CudaEventTimer

LOGGER = logging.getLogger(__name__)


class MonocularDepthBackend:
    """Depth Anything V2 backend; output is explicitly relative unless calibrated."""

    def __init__(self, config: ReconstructionConfig, device: str = "cuda") -> None:
        self.config = config
        self.requested_device = device
        self.device = "cpu"
        self.processor = None
        self.model = None
        self.last_gpu_ms = 0.0

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = self.requested_device if self.requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.config.model)
        dtype = torch.float16 if self.device.startswith("cuda") and self.config.fp16 else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(self.config.model, torch_dtype=dtype)
        self.model.to(self.device).eval()
        LOGGER.info("Loaded %s on %s", self.config.model, self.device)

    def warmup(self) -> None:
        dummy = np.full((self.config.input_size, self.config.input_size, 3), 127, dtype=np.uint8)
        packet = FramePacket(0, 0.0, time.perf_counter(), dummy[..., ::-1], dummy, 0.0, dummy.shape[1], dummy.shape[0])
        self.infer(packet)

    def infer(self, frame: FramePacket) -> PointCloudFrame:
        if self.model is None or self.processor is None:
            raise RuntimeError("Depth model has not been loaded")
        import torch

        start = time.perf_counter()
        resized_rgb = resize_for_pointmap(frame.rgb, self.config.input_size)
        inputs = self.processor(images=Image.fromarray(resized_rgb), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if self.device.startswith("cuda"):
            pixel_values = pixel_values.pin_memory().to(self.device, non_blocking=True)
        else:
            pixel_values = pixel_values.to(self.device)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.startswith("cuda") and self.config.fp16
            else contextlib.nullcontext()
        )
        timer = CudaEventTimer(self.device.startswith("cuda"))
        with torch.inference_mode(), autocast, timer:
            outputs = self.model(pixel_values=pixel_values)
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=resized_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        self.last_gpu_ms = timer.elapsed_ms
        inverse_depth = prediction.float().cpu().numpy()
        depth = relative_inverse_depth(inverse_depth)
        pointmap = depth_to_pointmap(depth)
        confidence = _depth_confidence(inverse_depth)
        points, colors, selected_confidence = voxel_downsample(
            pointmap,
            resized_rgb,
            confidence,
            voxel_size=self.config.voxel_size,
            max_points=self.config.max_points,
        )
        return PointCloudFrame(
            points=points,
            colors=colors,
            confidence=selected_confidence,
            pointmap=pointmap,
            frame_index=frame.frame_index,
            timestamp=frame.source_timestamp,
            anchor_frame_index=frame.frame_index,
            inference_ms=(time.perf_counter() - start) * 1000.0,
            valid=len(points) > 0,
            source="depth_anything_v2_relative",
            dense_confidence=confidence,
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _depth_confidence(inverse_depth: np.ndarray) -> np.ndarray:
    """Edge-aware confidence used only for filtering/display, not model certainty."""
    gx = cv2.Sobel(inverse_depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inverse_depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    scale = max(float(np.percentile(gradient, 90)), 1e-6)
    return np.exp(-gradient / scale).astype(np.float32)
