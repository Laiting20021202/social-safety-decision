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
        target_size: tuple[int, int] | None = None,
        crop_top_aspect_ratio: float | None = None,
        smooth_interpolate: bool = False,
    ) -> CachedVideo:
        scenario = frame_source.get_scenario(scenario_id)
        path = self._video_path(
            scenario,
            target_fps=target_fps,
            target_size=target_size,
            crop_top_aspect_ratio=crop_top_aspect_ratio,
            smooth_interpolate=smooth_interpolate,
        )
        if path.exists() and path.stat().st_size > 0:
            return CachedVideo(
                path=path,
                fps=target_fps,
                source="cached_smooth_mp4" if smooth_interpolate else "cached_mp4",
                generated=False,
            )
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
        vf = _video_filter(
            target_size,
            crop_top_aspect_ratio=crop_top_aspect_ratio,
            smooth_interpolate=smooth_interpolate,
            target_fps=target_fps,
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
            vf,
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
            source="generated_smooth_mp4" if smooth_interpolate else "generated_from_frames",
            generated=True,
        )

    def _video_path(
        self,
        scenario: ScenarioInfo,
        *,
        target_fps: float,
        target_size: tuple[int, int] | None,
        crop_top_aspect_ratio: float | None,
        smooth_interpolate: bool,
    ) -> Path:
        dataset = _safe_path_part(scenario.dataset_name)
        scenario_name = _safe_path_part(scenario.scenario_id)
        revision = _safe_path_part(scenario.dataset_revision or "local")
        suffix = f"{target_fps:g}fps"
        if crop_top_aspect_ratio is not None:
            suffix = f"top{crop_top_aspect_ratio:.3g}_{suffix}"
        if smooth_interpolate:
            suffix = f"smooth_{suffix}"
        if target_size is not None:
            suffix = f"{target_size[0]}x{target_size[1]}_{suffix}"
        return self.root / dataset / revision / f"{scenario_name}_{suffix}.mp4"


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


def _video_filter(
    target_size: tuple[int, int] | None,
    *,
    crop_top_aspect_ratio: float | None = None,
    smooth_interpolate: bool = False,
    target_fps: float = 25.0,
) -> str:
    filters: list[str] = []
    if crop_top_aspect_ratio is not None:
        filters.append(f"crop=iw:min(ih\\,iw/{crop_top_aspect_ratio:.8g}):0:0")
    if target_size is not None:
        width, height = target_size
        filters.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    if smooth_interpolate:
        filters.append(
            f"minterpolate=fps={target_fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
    filters.append("format=yuv420p")
    return ",".join(filters)


def _safe_path_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
