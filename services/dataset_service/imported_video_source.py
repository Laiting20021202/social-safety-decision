from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from packages.common_models import DatasetInfo, FramePacket, ScenarioInfo
from packages.frame_sources.base import FrameSource
from services.dataset_service.video_imports import ImportedVideoRecord


class ImportedVideoFrameSource(FrameSource):
    def __init__(
        self,
        record: ImportedVideoRecord,
        *,
        cache_root: str | Path,
        analysis_fps: float = 5.0,
    ) -> None:
        self.record = record
        self.cache_root = Path(cache_root)
        self.analysis_fps = max(0.1, float(analysis_fps))
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            dataset_id=f"imported-video/{self.record.video_id}",
            name=self.record.dataset_name,
            revision=self.record.video_hash,
            source_url=self.record.path,
            cached=True,
            metadata={
                "source": "local_mp4",
                "native_fps": self.record.native_fps,
                "analysis_fps": self.analysis_fps,
            },
        )

    def list_scenarios(self) -> list[ScenarioInfo]:
        return [self.get_scenario(self.record.video_id)]

    def get_scenario(self, scenario_id: str) -> ScenarioInfo:
        if scenario_id != self.record.video_id:
            raise FileNotFoundError(f"Imported video scenario not found: {scenario_id}")
        frame_count = max(1, int(self.record.duration_sec * self.analysis_fps) + 1)
        preview_timestamp = _preview_timestamp(self.record.duration_sec)
        return ScenarioInfo(
            scenario_id=self.record.video_id,
            dataset_name=self.record.dataset_name,
            dataset_revision=self.record.video_hash,
            split="local_mp4",
            frame_count=frame_count,
            duration_sec=self.record.duration_sec,
            image_width=self.record.width,
            image_height=self.record.height,
            first_frame_index=0,
            last_frame_index=frame_count - 1,
            metadata={
                "name": self.record.name,
                "video_id": self.record.video_id,
                "dataset_name": self.record.dataset_name,
                "source": self.record.source,
                "timestamp_source": "mp4_analysis_fps",
                "video_path": self.record.path,
                "native_fps": self.record.native_fps,
                "analysis_fps": self.analysis_fps,
                "preview_timestamp_sec": preview_timestamp,
            },
        )

    def get_frame(self, scenario_id: str, frame_index: int) -> FramePacket:
        scenario = self.get_scenario(scenario_id)
        if frame_index < 0 or frame_index >= scenario.frame_count:
            raise IndexError(f"Frame {frame_index} outside imported video {scenario_id}")
        timestamp = min(self.record.duration_sec, frame_index / self.analysis_fps)
        return FramePacket(
            source_type="local_video",
            dataset_name=self.record.dataset_name,
            dataset_revision=self.record.video_hash,
            split="local_mp4",
            scenario_id=scenario_id,
            frame_index=frame_index,
            timestamp_sec=timestamp,
            original_timestamp=timestamp,
            fps=self.analysis_fps,
            image_width=self.record.width,
            image_height=self.record.height,
            image_reference=f"/videos/imports/{self.record.video_id}/frames/{frame_index}/image",
            metadata={
                "name": self.record.name,
                "video_id": self.record.video_id,
                "dataset_name": self.record.dataset_name,
                "source": self.record.source,
                "video_path": self.record.path,
                "analysis_fps": self.analysis_fps,
                "native_fps": self.record.native_fps,
                "preview_timestamp_sec": _preview_timestamp(self.record.duration_sec),
            },
        )

    def get_frame_image_path(self, scenario_id: str, frame_index: int) -> Path:
        frame = self.get_frame(scenario_id, frame_index)
        path = self.cache_root / f"{frame_index:06d}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return path
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for Local MP4 lightweight analysis")
        timestamp = min(max(0.0, frame.timestamp_sec), max(0.0, self.record.duration_sec - 0.001))
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            self.record.path,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not path.exists():
            raise RuntimeError(
                completed.stderr.strip() or f"ffmpeg failed to extract frame {frame_index}"
            )
        return path


def _preview_timestamp(duration_sec: float) -> float:
    if duration_sec <= 1.0:
        return 0.0
    return min(max(duration_sec * 0.25, 0.0), 10.0)
