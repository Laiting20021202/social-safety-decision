from __future__ import annotations

import contextlib
import logging
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.pointcloud import voxel_downsample
from realtime_safety.types import FramePacket, PointCloudFrame
from realtime_safety.utils.timing import CudaEventTimer

LOGGER = logging.getLogger(__name__)


class St4RTrackAdapter:
    """In-memory two-frame wrapper around the official St4RTrack model.

    It bypasses upstream's directory/video loader and NPY export. Upstream code is
    discovered via ``st4rtrack_path`` and remains an optional external dependency.
    """

    def __init__(self, config: ReconstructionConfig, device: str = "cuda") -> None:
        self.config = config
        self.requested_device = device
        self.device = "cpu"
        self.model = None
        self._anchor: FramePacket | None = None
        self._anchor_view: dict | None = None
        self.last_gpu_ms = 0.0

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch

        root = Path(self.config.st4rtrack_path or "third_party/St4RTrack").expanduser().resolve()
        if not (root / "dust3r" / "model.py").is_file():
            raise FileNotFoundError(
                f"St4RTrack not found at {root}. Run scripts/download_models.sh --st4rtrack or use hybrid/fast_depth."
            )
        sys.path.insert(0, str(root)) if str(root) not in sys.path else None
        # Upstream imports its evaluation-only ``evo`` helper at model import
        # time. The live adapter never evaluates camera trajectories, so avoid
        # requiring that unrelated package in the lightweight viewer runtime.
        try:
            import evo  # noqa: F401
        except ImportError:
            vo_eval = types.ModuleType("dust3r.utils.vo_eval")

            def _evaluation_only(*_args, **_kwargs):
                raise RuntimeError("Trajectory export requires the optional St4RTrack evaluation dependencies")

            vo_eval.save_trajectory_tum_format = _evaluation_only
            sys.modules.setdefault("dust3r.utils.vo_eval", vo_eval)
        from dust3r.model import AsymmetricCroCo3DStereo

        self.device = self.requested_device if self.requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        checkpoint = self.config.st4rtrack_checkpoint
        model_source = checkpoint if checkpoint and Path(checkpoint).is_file() else "yupengchengg147/St4RTrack"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_source).to(self.device).eval()
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        LOGGER.info("Loaded St4RTrack from %s on %s", model_source, self.device)

    def warmup(self) -> None:
        size = self.config.input_size
        rgb = np.full((size, size, 3), 127, dtype=np.uint8)
        packet = FramePacket(0, 0.0, time.perf_counter(), rgb[..., ::-1], rgb, 0.0, size, size)
        self.set_anchor(packet)
        self.infer(packet, packet)

    def set_anchor(self, frame: FramePacket) -> None:
        self._anchor = frame
        self._anchor_view, _ = self._make_view(frame.rgb, frame.frame_index)

    def infer(self, anchor_frame: FramePacket | None, current_frame: FramePacket) -> PointCloudFrame:
        if self.model is None:
            raise RuntimeError("St4RTrack model has not been loaded")
        if anchor_frame is not None and (self._anchor is None or anchor_frame.frame_index != self._anchor.frame_index):
            self.set_anchor(anchor_frame)
        if self._anchor_view is None or self._anchor is None:
            self.set_anchor(current_frame)
        import torch

        start = time.perf_counter()
        current_view, current_rgb = self._make_view(current_frame.rgb, current_frame.frame_index)
        anchor_view = {key: value for key, value in self._anchor_view.items()}
        anchor_view["img"] = anchor_view["img"].to(self.device, non_blocking=True)
        anchor_view["true_shape"] = anchor_view["true_shape"].to(self.device, non_blocking=True)
        current_view["img"] = current_view["img"].to(self.device, non_blocking=True)
        current_view["true_shape"] = current_view["true_shape"].to(self.device, non_blocking=True)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.startswith("cuda") and self.config.fp16
            else contextlib.nullcontext()
        )
        timer = CudaEventTimer(self.device.startswith("cuda"))
        with torch.inference_mode(), autocast, timer:
            pred_tracking, pred_reconstruction = self.model(anchor_view, current_view)
        self.last_gpu_ms = timer.elapsed_ms
        camera_pointmap = pred_reconstruction["pts3d_in_other_view"][0].float().cpu().numpy()
        pointmap = np.stack((camera_pointmap[..., 0], camera_pointmap[..., 2], -camera_pointmap[..., 1]), axis=-1)
        reconstruction_conf = pred_reconstruction["conf"][0].float().cpu().numpy()
        camera_tracking = pred_tracking["pts3d"][0].float().cpu().numpy()
        tracking_pointmap = np.stack((camera_tracking[..., 0], camera_tracking[..., 2], -camera_tracking[..., 1]), axis=-1)
        tracking_conf = pred_tracking["conf"][0].float().cpu().numpy()
        finite = np.isfinite(pointmap).all(axis=-1)
        display_confidence = max(
            self.config.confidence_threshold,
            self.config.display_confidence_threshold,
        )
        valid = finite & (reconstruction_conf >= display_confidence)
        if self.config.filter_sky:
            sky = _connected_sky_mask(current_rgb)
            valid &= ~sky
            # A color-only mask cannot catch all gray/overexposed skies. Reject
            # upper-image points whose predicted depth is implausibly nearer
            # than the robust lower-image scene scale.
            height = pointmap.shape[0]
            lower = finite[int(height * 0.62) :] & (pointmap[int(height * 0.62) :, :, 1] > 0.05)
            lower_depth = pointmap[int(height * 0.62) :, :, 1][lower]
            if len(lower_depth) > 32:
                near_cutoff = 0.35 * float(np.median(lower_depth))
                rows = np.arange(height)[:, None]
                invalid_near_upper = (rows < height * 0.58) & (pointmap[..., 1] < near_cutoff)
                valid &= ~invalid_near_upper
        points, colors, confidence = voxel_downsample(
            pointmap[valid], current_rgb[valid], reconstruction_conf[valid], self.config.voxel_size, self.config.max_points
        )
        tracking_valid = np.isfinite(tracking_pointmap).all(axis=-1) & (tracking_conf >= self.config.confidence_threshold)
        tracking_points = tracking_pointmap[tracking_valid]
        if len(tracking_points) > self.config.max_points // 4:
            indices = np.linspace(0, len(tracking_points) - 1, self.config.max_points // 4, dtype=np.int64)
            tracking_points = tracking_points[indices]
        return PointCloudFrame(
            points=points,
            colors=colors,
            confidence=confidence,
            pointmap=pointmap,
            frame_index=current_frame.frame_index,
            timestamp=current_frame.source_timestamp,
            anchor_frame_index=self._anchor.frame_index,
            inference_ms=(time.perf_counter() - start) * 1000.0,
            valid=len(points) > 0,
            source="st4rtrack",
            tracking_points=tracking_points.astype(np.float32),
            dense_confidence=reconstruction_conf,
        )

    def reset(self) -> None:
        self._anchor = None
        self._anchor_view = None

    def close(self) -> None:
        self.reset()
        self.model = None
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _make_view(self, rgb: np.ndarray, frame_index: int) -> tuple[dict, np.ndarray]:
        import torch

        size = self.config.input_size
        height, width = rgb.shape[:2]
        if size == 224:
            scale = size / min(height, width)
            resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            y0 = (resized.shape[0] - size) // 2
            x0 = (resized.shape[1] - size) // 2
            resized = resized[y0 : y0 + size, x0 : x0 + size]
        else:
            scale = size / max(height, width)
            target_w = max(16, int(round(width * scale / 16)) * 16)
            target_h = max(16, int(round(height * scale / 16)) * 16)
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(0)
        if self.device.startswith("cuda"):
            tensor = tensor.pin_memory()
        true_shape = torch.tensor([[resized.shape[0], resized.shape[1]]], dtype=torch.int32)
        return {
            "img": tensor,
            "true_shape": true_shape,
            "idx": frame_index,
            "instance": f"memory_frame_{frame_index}",
        }, resized


def _connected_sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Find blue or bright low-texture regions connected to the image top."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    rows = np.arange(height)[:, None]
    upper = rows < height * 0.62
    blue = (
        upper
        & (hsv[..., 0] >= 85)
        & (hsv[..., 0] <= 132)
        & (hsv[..., 1] >= 38)
        & (hsv[..., 2] >= 65)
    )
    bright_smooth = upper & (hsv[..., 1] < 45) & (hsv[..., 2] > 165) & (gradient < 18.0)
    candidates = (blue | bright_smooth).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    result = np.zeros((height, width), dtype=bool)
    top_labels = set(np.unique(labels[: max(2, height // 40)]).tolist())
    minimum_area = max(32, int(height * width * 0.002))
    for label in range(1, count):
        if label in top_labels and int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            result |= labels == label
    return cv2.dilate(result.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
