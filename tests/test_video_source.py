from pathlib import Path

import cv2
import numpy as np

from realtime_safety.pipeline.video_source import VideoSource


def test_file_timestamps_start_at_zero_and_follow_source_fps(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for value in range(3):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    source = VideoSource(str(path))
    source.open()
    frames = [source.read() for _ in range(3)]
    source.close()
    assert all(frame is not None for frame in frames)
    timestamps = [frame.source_timestamp for frame in frames if frame is not None]
    assert timestamps[0] == 0.0
    np.testing.assert_allclose(np.diff(timestamps), [0.1, 0.1], atol=0.02)
