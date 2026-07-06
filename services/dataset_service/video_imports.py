from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImportedVideoRecord:
    video_id: str
    name: str
    dataset_name: str
    path: str
    native_fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    video_hash: str
    source: str
    video_reference: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "path": self.path,
            "video_reference": f"/videos/imports/{self.video_id}/video",
            "native_fps": self.native_fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
            "video_hash": self.video_hash,
            "source": self.source,
        }


class VideoImportStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.video_root = self.root / "files"
        self.index_path = self.root / "index.json"
        self.video_root.mkdir(parents=True, exist_ok=True)

    def list_records(self) -> list[ImportedVideoRecord]:
        return [
            ImportedVideoRecord(**item)
            for item in self._load_index().values()
            if isinstance(item, dict)
        ]

    def get_record(self, video_id: str) -> ImportedVideoRecord:
        item = self._load_index().get(video_id)
        if not isinstance(item, dict):
            raise FileNotFoundError(f"Imported video not found: {video_id}")
        return ImportedVideoRecord(**item)

    def import_file(
        self,
        source_path: str | Path,
        *,
        name: str | None = None,
        dataset_name: str | None = None,
        source: str = "path",
    ) -> ImportedVideoRecord:
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MP4 path does not exist: {path}")
        if path.suffix.lower() != ".mp4":
            raise ValueError("Only .mp4 files can be imported as Local Continuous Video.")
        info = probe_mp4(path)
        video_hash = sha256_file(path)
        video_id = video_hash[:16]
        target_path = self.video_root / f"{video_id}.mp4"
        if not target_path.exists() or sha256_file(target_path) != video_hash:
            shutil.copy2(path, target_path)
        record = ImportedVideoRecord(
            video_id=video_id,
            name=name or path.stem,
            dataset_name=dataset_name or "Local Continuous Video",
            path=str(target_path),
            native_fps=info["native_fps"],
            frame_count=info["frame_count"],
            duration_sec=info["duration_sec"],
            width=info["width"],
            height=info["height"],
            video_hash=video_hash,
            source=source,
        )
        self._save_record(record)
        return record

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_record(self, record: ImportedVideoRecord) -> None:
        index = self._load_index()
        index[record.video_id] = record.as_dict()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def probe_mp4(path: str | Path) -> dict[str, int | float]:
    resolved = Path(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(resolved),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe failed for {resolved}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"No video stream found in {resolved}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"Invalid ffprobe stream payload for {resolved}")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(str(stream.get("avg_frame_rate") or "0/1"))
    duration = _float_or_zero(stream.get("duration"))
    if duration <= 0:
        format_payload = payload.get("format")
        if isinstance(format_payload, dict):
            duration = _float_or_zero(format_payload.get("duration"))
    frame_count = _int_or_zero(stream.get("nb_frames"))
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0 or duration <= 0:
        raise RuntimeError(
            "MP4 probe returned invalid metadata: "
            f"width={width}, height={height}, fps={fps}, frames={frame_count}, duration={duration}"
        )
    return {
        "native_fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "width": width,
        "height": height,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_rate(value: str) -> float:
    if "/" not in value:
        return _float_or_zero(value)
    numerator, denominator = value.split("/", 1)
    denominator_float = _float_or_zero(denominator)
    if denominator_float <= 0:
        return 0.0
    return _float_or_zero(numerator) / denominator_float


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
