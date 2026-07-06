from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from social_bev.types import PipelineResult
from social_bev.utils import ensure_dir, safe_cv2_imwrite, to_jsonable


class OutputWriter:
    """Write video, JSONL, and occupancy grids incrementally."""

    def __init__(
        self,
        video_path: str | Path | None,
        jsonl_path: str | Path,
        occupancy_dir: str | Path,
        fps: float,
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.jsonl_path = Path(jsonl_path)
        self.occupancy_dir = Path(occupancy_dir)
        self.fps = float(fps) if fps and fps > 1e-3 else 20.0
        self._writer: cv2.VideoWriter | None = None
        ensure_dir(self.jsonl_path.parent)
        ensure_dir(self.occupancy_dir)
        self.jsonl_path.write_text("", encoding="utf-8")

    def write(self, result: PipelineResult) -> None:
        self._write_video(result.visualization)
        self._write_json(result)
        self._write_occupancy(result)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _write_video(self, frame: np.ndarray) -> None:
        if self.video_path is None:
            return
        if self._writer is None:
            ensure_dir(self.video_path.parent)
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise IOError(f"Failed to open output video: {self.video_path}")
        self._writer.write(frame)

    def _write_json(self, result: PipelineResult) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(result.to_json_dict()), ensure_ascii=False) + "\n")

    def _write_occupancy(self, result: PipelineResult) -> None:
        stem = f"frame_{result.frame_index + 1:06d}"
        grid = result.bev.occupancy_grid.astype(np.int16)
        np.save(self.occupancy_dir / f"{stem}.npy", grid)
        png = grid.copy()
        png[png < 0] = 127
        png = np.clip(png, 0, 255).astype(np.uint8)
        safe_cv2_imwrite(self.occupancy_dir / f"{stem}.png", png)

    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

