import time

import numpy as np

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.video_depth import VideoDepthBackend
from realtime_safety.types import FramePacket


class _FakeStreamingDepth:
    def __init__(self) -> None:
        self.transform = object()
        self.frame_id_list = [1]
        self.frame_cache_list = [object()]
        self.id = 1

    def infer_video_depth_one(self, frame, input_size, device, fp32):
        assert input_size == 280
        assert device == "cpu"
        assert fp32
        return np.full(frame.shape[:2], 2.0, dtype=np.float32)


def _packet() -> FramePacket:
    rgb = np.full((64, 96, 3), 120, dtype=np.uint8)
    return FramePacket(3, 0.3, time.perf_counter(), rgb[..., ::-1], rgb, 30.0, 96, 64)


def test_metric_video_depth_uses_camera_intrinsics_for_pointmap() -> None:
    config = ReconstructionConfig(
        depth_mode="video_depth",
        input_size=280,
        max_points=20_000,
        voxel_size=0.0,
        focal_length_x=100.0,
        focal_length_y=80.0,
        principal_point_x=48.0,
        principal_point_y=32.0,
    )
    backend = VideoDepthBackend(config, device="cpu")
    backend.model = _FakeStreamingDepth()
    output = backend.infer(_packet())

    assert output.source == "video_depth_anything_metric"
    assert output.pointmap.shape == (64, 96, 3)
    assert np.allclose(output.pointmap[32, 48], (0.0, 2.0, 0.0))
    assert np.isclose(output.pointmap[32, 58, 0], 0.2)
    assert np.isclose(output.pointmap[24, 48, 2], 0.2)
    assert output.valid


def test_video_depth_reset_discards_temporal_state() -> None:
    backend = VideoDepthBackend(ReconstructionConfig(), device="cpu")
    model = _FakeStreamingDepth()
    backend.model = model
    backend.reset()
    assert model.transform is None
    assert model.frame_id_list == []
    assert model.frame_cache_list == []
    assert model.id == -1
