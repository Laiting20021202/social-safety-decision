from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from social_bev.frame_source import FrameSource


def test_image_directory_frame_source(tmp_path: Path) -> None:
    for idx in range(3):
        image = np.full((32, 48, 3), idx * 30, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{idx:03d}.jpg"), image)
    frames = list(FrameSource(tmp_path))
    assert len(frames) == 3
    assert frames[0].image.shape == (32, 48, 3)


def test_video_frame_source(tmp_path: Path) -> None:
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (48, 32))
    assert writer.isOpened()
    for idx in range(4):
        writer.write(np.full((32, 48, 3), idx * 40, dtype=np.uint8))
    writer.release()
    frames = list(FrameSource(video))
    assert len(frames) == 4
    assert abs(FrameSource(video).default_fps - 20.0) < 1e-6

