from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from packages.common_models import ScenarioInfo
from packages.frame_sources import FrameSource


@dataclass(frozen=True)
class CachedVideo:
    path: Path
    fps: float
    source: str
    generated: bool


class VideoCache:
    def __init__(self, root: str | Path = "outputs/video_cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_mp4(
        self,
        frame_source: FrameSource,
        scenario_id: str,
        target_fps: float = 25.0,
    ) -> CachedVideo:
        scenario = frame_source.get_scenario(scenario_id)
        path = self._video_path(scenario)
        if path.exists() and path.stat().st_size > 0:
            return CachedVideo(path=path, fps=target_fps, source="cached_mp4", generated=False)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to build cached MP4 playback assets")
        manifest = path.with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        interval_sec = _frame_interval_sec(scenario)
        manifest.write_text(
            _concat_manifest(frame_source, scenario, interval_sec),
            encoding="utf-8",
        )
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-vf",
            "format=yuv420p",
            "-r",
            f"{target_fps:g}",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(command, check=True)
        return CachedVideo(
            path=path,
            fps=target_fps,
            source="generated_from_frames",
            generated=True,
        )

    def _video_path(self, scenario: ScenarioInfo) -> Path:
        dataset = _safe_path_part(scenario.dataset_name)
        scenario_name = _safe_path_part(scenario.scenario_id)
        revision = _safe_path_part(scenario.dataset_revision or "local")
        return self.root / dataset / revision / f"{scenario_name}.mp4"


def _frame_interval_sec(scenario: ScenarioInfo) -> float:
    if scenario.frame_count > 1 and scenario.duration_sec > 0:
        return scenario.duration_sec / (scenario.frame_count - 1)
    metadata_interval = scenario.metadata.get("virtual_frame_interval_sec")
    if isinstance(metadata_interval, int | float) and metadata_interval > 0:
        return float(metadata_interval)
    return 1.0 / 25.0


def _concat_manifest(
    frame_source: FrameSource,
    scenario: ScenarioInfo,
    interval_sec: float,
) -> str:
    lines: list[str] = []
    last_path: Path | None = None
    for frame_index in range(scenario.frame_count):
        path = Path(frame_source.get_frame_image_path(scenario.scenario_id, frame_index))
        last_path = path
        lines.append(f"file '{_escape_concat_path(path)}'")
        lines.append(f"duration {interval_sec:.6f}")
    if last_path is not None:
        lines.append(f"file '{_escape_concat_path(last_path)}'")
    return "\n".join(lines) + "\n"


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", r"'\''")


def _safe_path_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
