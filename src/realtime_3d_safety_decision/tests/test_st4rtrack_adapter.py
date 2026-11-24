import time

import numpy as np
import torch

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.st4rtrack_adapter import St4RTrackAdapter, _connected_sky_mask
from realtime_safety.types import FramePacket


class _PairModel:
    def __call__(self, anchor: dict, current: dict):
        _, _, height, width = current["img"].shape
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        camera_points = torch.stack((xx / width, yy / height, torch.ones_like(xx)), dim=-1).float().unsqueeze(0)
        confidence = torch.ones((1, height, width), dtype=torch.float32)
        return {"pts3d": camera_points, "conf": confidence}, {"pts3d_in_other_view": camera_points, "conf": confidence}


def _packet(index: int) -> FramePacket:
    rgb = np.full((64, 96, 3), 100 + index, dtype=np.uint8)
    return FramePacket(index, index * 0.1, time.perf_counter(), rgb[..., ::-1], rgb, 10.0, 96, 64)


def test_adapter_accepts_memory_frames_and_returns_pointmaps() -> None:
    config = ReconstructionConfig(input_size=224, max_points=1000, voxel_size=0.02)
    adapter = St4RTrackAdapter(config, device="cpu")
    adapter.model = _PairModel()
    anchor, current = _packet(0), _packet(1)
    adapter.set_anchor(anchor)
    output = adapter.infer(anchor, current)
    assert output.source == "st4rtrack"
    assert output.frame_index == 1
    assert output.anchor_frame_index == 0
    assert output.pointmap.shape == (224, 224, 3)
    assert 0 < len(output.points) <= config.max_points
    assert output.tracking_points is not None


def test_connected_sky_mask_removes_top_sky_but_keeps_ground() -> None:
    rgb = np.zeros((100, 160, 3), dtype=np.uint8)
    rgb[:55] = (65, 145, 230)
    rng = np.random.default_rng(3)
    rgb[55:] = rng.integers((30, 60, 20), (110, 155, 90), size=(45, 160, 3), dtype=np.uint8)
    mask = _connected_sky_mask(rgb)
    assert mask[:50].mean() > 0.95
    assert mask[65:].mean() < 0.01
