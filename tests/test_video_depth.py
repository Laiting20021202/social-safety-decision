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


class _ShapeLockedStreamingDepth(_FakeStreamingDepth):
    def __init__(self) -> None:
        super().__init__()
        self.transform = None
        self.initialized_shapes: list[tuple[int, int]] = []

    def infer_video_depth_one(self, frame, input_size, device, fp32):
        shape = tuple(frame.shape[:2])
        if self.transform is None:
            self.transform = object()
            self.frame_height, self.frame_width = shape
            self.initialized_shapes.append(shape)
        else:
            assert shape == (self.frame_height, self.frame_width)
        return np.full(shape, 2.0, dtype=np.float32)


class _OutlierStreamingDepth(_FakeStreamingDepth):
    def infer_video_depth_one(self, frame, input_size, device, fp32):
        depth = np.full(frame.shape[:2], 1.0, dtype=np.float32)
        depth[:8, :8] = 50.0
        return depth


def _packet(height: int = 64, width: int = 96) -> FramePacket:
    rgb = np.full((height, width, 3), 120, dtype=np.uint8)
    return FramePacket(3, 0.3, time.perf_counter(), rgb[..., ::-1], rgb, 30.0, width, height)


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


def test_metric_video_depth_calibrates_fixed_reference_before_projection() -> None:
    config = ReconstructionConfig(
        depth_mode="video_depth",
        input_size=280,
        max_points=20_000,
        voxel_size=0.0,
        metric_reference_depth_m=0.4,
        metric_reference_roi=(0.4, 0.4, 0.6, 0.6),
        metric_reference_percentile=50.0,
        metric_reference_warmup_frames=1,
    )
    backend = VideoDepthBackend(config, device="cpu")
    backend.model = _FakeStreamingDepth()
    output = backend.infer(_packet())

    assert np.isclose(output.metric_scale, 0.2)
    assert np.isclose(output.reference_observed_depth, 2.0)
    assert np.isclose(output.pointmap[32, 48, 1], 0.4)
    assert output.source == "video_depth_anything_metric_reference_calibrated"


def test_metric_calibration_does_not_readmit_large_raw_depth_outliers() -> None:
    config = ReconstructionConfig(
        depth_mode="video_depth",
        input_size=280,
        max_points=20_000,
        voxel_size=0.0,
        max_relative_depth=30.0,
        max_metric_depth_m=4.0,
        metric_reference_depth_m=0.4,
        metric_reference_roi=(0.4, 0.4, 0.6, 0.6),
        metric_reference_percentile=50.0,
        metric_reference_warmup_frames=1,
    )
    backend = VideoDepthBackend(config, device="cpu")
    backend.model = _OutlierStreamingDepth()
    output = backend.infer(_packet())

    assert np.isclose(output.metric_scale, 0.4)
    assert len(output.points) == 64 * 96 - 8 * 8
    assert output.points[:, 1].max() < 4.0


def test_video_depth_resets_temporal_state_when_stream_resolution_changes() -> None:
    backend = VideoDepthBackend(ReconstructionConfig(input_size=280, voxel_size=0.0), device="cpu")
    model = _ShapeLockedStreamingDepth()
    backend.model = model

    first = backend.infer(_packet(64, 96))
    second = backend.infer(_packet(48, 80))

    assert first.pointmap.shape == (64, 96, 3)
    assert second.pointmap.shape == (48, 80, 3)
    assert model.initialized_shapes == [(64, 96), (48, 80)]
