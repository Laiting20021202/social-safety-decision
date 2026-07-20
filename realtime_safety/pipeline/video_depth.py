from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.pointcloud import depth_to_pointmap, resize_for_pointmap, voxel_downsample
from realtime_safety.types import FramePacket, PointCloudFrame
from realtime_safety.utils.timing import CudaEventTimer

LOGGER = logging.getLogger(__name__)


class VideoDepthBackend:
    """Streaming Metric Video Depth Anything Small backend.

    Unlike an independent monocular estimate on every frame, the upstream
    streaming model reuses temporal-attention state from earlier frames. This
    makes it a better fit for moving hands and other short occlusions while
    preserving one-frame-at-a-time live inference.
    """

    def __init__(self, config: ReconstructionConfig, device: str = "cuda") -> None:
        self.config = config
        self.requested_device = device
        self.device = "cpu"
        self.model = None
        self.last_gpu_ms = 0.0

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        root = Path(self.config.video_depth_path or "third_party/Video-Depth-Anything").expanduser().resolve()
        module = root / "video_depth_anything" / "video_depth_stream.py"
        if not module.is_file():
            raise FileNotFoundError(
                f"Video Depth Anything not found at {root}. "
                "Run scripts/download_models.sh --viewer or select --depth-mode st4rtrack."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from video_depth_anything.video_depth_stream import VideoDepthAnything

        self.device = (
            self.requested_device
            if self.requested_device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.model = VideoDepthAnything(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        checkpoint = self.config.video_depth_checkpoint
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else Path(
                hf_hub_download(
                    "depth-anything/Metric-Video-Depth-Anything-Small",
                    "metric_video_depth_anything_vits.pth",
                )
            )
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        LOGGER.info("Loaded Metric Video Depth Anything Small from %s on %s", checkpoint_path, self.device)

    def warmup(self) -> None:
        # The first streaming call creates the temporal cache. Keep that cache
        # out of the real source, then reset it so two unrelated scenes cannot
        # contaminate one another.
        size = max(56, min(self.config.input_size, 280))
        dummy = np.full((size, round(size * 4 / 3), 3), 127, dtype=np.uint8)
        packet = FramePacket(
            0,
            0.0,
            time.perf_counter(),
            dummy[..., ::-1],
            dummy,
            0.0,
            dummy.shape[1],
            dummy.shape[0],
        )
        self.infer(packet)
        self.reset()

    def infer(self, frame: FramePacket) -> PointCloudFrame:
        if self.model is None:
            raise RuntimeError("Video depth model has not been loaded")

        start = time.perf_counter()
        timer = CudaEventTimer(self.device.startswith("cuda"))
        with timer:
            depth = self.model.infer_video_depth_one(
                frame.rgb,
                input_size=self.config.input_size,
                device=self.device,
                fp32=not self.config.fp16 or not self.device.startswith("cuda"),
            )
        self.last_gpu_ms = timer.elapsed_ms
        depth = np.asarray(depth, dtype=np.float32)
        pointmap_rgb = resize_for_pointmap(frame.rgb, self.config.input_size)
        if depth.shape != pointmap_rgb.shape[:2]:
            depth = cv2.resize(
                depth,
                (pointmap_rgb.shape[1], pointmap_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        focal_x, focal_y, center = self._camera_intrinsics(depth.shape, frame.rgb.shape[:2])
        pointmap = depth_to_pointmap(depth, (focal_x, focal_y), center)
        confidence = _temporal_depth_confidence(depth)
        valid = (
            np.isfinite(depth)
            & (depth > 0.05)
            & (depth < self.config.max_relative_depth)
        )
        points, colors, selected_confidence = voxel_downsample(
            pointmap[valid],
            pointmap_rgb[valid],
            confidence[valid],
            self.config.voxel_size,
            self.config.max_points,
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
            source="video_depth_anything_metric",
            dense_confidence=confidence,
        )

    def reset(self) -> None:
        if self.model is None:
            return
        # These are the state variables defined by upstream's streaming API.
        self.model.transform = None
        self.model.frame_id_list = []
        self.model.frame_cache_list = []
        self.model.id = -1

    def close(self) -> None:
        self.reset()
        self.model = None
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _camera_intrinsics(
        self,
        shape: tuple[int, int],
        source_shape: tuple[int, int],
    ) -> tuple[float, float, tuple[float, float]]:
        height, width = shape
        source_height, source_width = source_shape
        scale_x = width / source_width
        scale_y = height / source_height
        fallback = 0.85 * max(width, height)
        configured_x = self.config.focal_length_x or self.config.focal_length_y
        configured_y = self.config.focal_length_y or self.config.focal_length_x
        focal_x = float(configured_x * scale_x if configured_x is not None else fallback)
        focal_y = float(configured_y * scale_y if configured_y is not None else fallback)
        center = (
            float(self.config.principal_point_x) * scale_x
            if self.config.principal_point_x is not None
            else (width - 1) * 0.5,
            float(self.config.principal_point_y) * scale_y
            if self.config.principal_point_y is not None
            else (height - 1) * 0.5,
        )
        return focal_x, focal_y, center


def _temporal_depth_confidence(depth: np.ndarray) -> np.ndarray:
    """Edge-aware sampling weight; model time consistency supplies stability."""
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    scale = max(float(np.percentile(gradient, 90)), 1e-6)
    return np.exp(-gradient / scale).astype(np.float32)
