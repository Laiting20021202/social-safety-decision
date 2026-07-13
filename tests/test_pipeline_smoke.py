import os
from pathlib import Path

import pytest

from realtime_safety.config import load_config
from realtime_safety.scheduler import RealtimePipeline


@pytest.mark.model
def test_real_model_pipeline_smoke_when_video_is_supplied() -> None:
    source = os.environ.get("SAFETY_SMOKE_VIDEO")
    if not source or not Path(source).is_file():
        pytest.skip("Set SAFETY_SMOKE_VIDEO to run the neural-model/CUDA smoke test")
    config = load_config("realtime_fast")
    config.reconstruction.depth_mode = "fast_depth"
    pipeline = RealtimePipeline(config)
    pipeline.start_workers()
    pipeline.start_source(source, max_frames=30)
    assert pipeline.wait_until_source_done(20.0)
    snapshot = pipeline.gui_state.read()
    pipeline.close()
    assert snapshot.detections
    assert snapshot.pointcloud is not None and snapshot.pointcloud.valid
    assert snapshot.safety is not None
    assert snapshot.safety.danger_zones
    import torch

    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() < 1024 * 1024
