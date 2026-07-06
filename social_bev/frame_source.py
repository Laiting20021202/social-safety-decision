from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from social_bev.types import Frame
from social_bev.utils import image_files_in_directory


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class FrameSource:
    """Iterate frames from video, image, image directory, webcam, or SCAND sample directory."""

    def __init__(self, source: str | int | Path, default_fps: float = 20.0, stride: int = 1) -> None:
        self.source = source
        self.default_fps = float(default_fps)
        self.stride = max(1, int(stride))
        self.kind = self._detect_kind(source)
        self.fps = self.default_fps
        self.frame_count: int | None = None

    def __iter__(self) -> Iterator[Frame]:
        if self.kind == "directory":
            yield from self._iter_directory()
        elif self.kind == "image":
            yield from self._iter_image()
        else:
            yield from self._iter_capture()

    def _detect_kind(self, source: str | int | Path) -> str:
        if isinstance(source, int):
            return "webcam"
        source_str = str(source)
        if source_str.isdigit():
            return "webcam"
        path = Path(source_str)
        if path.is_dir():
            return "directory"
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        raise ValueError(f"Unsupported input source: {source}")

    def _iter_directory(self) -> Iterator[Frame]:
        directory = Path(str(self.source))
        image_dir = directory / "images" if (directory / "images").is_dir() else directory
        files = image_files_in_directory(image_dir)
        self.frame_count = len(files)
        self.fps = self.default_fps
        output_index = 0
        for original_index, path in enumerate(files):
            if original_index % self.stride != 0:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield Frame(
                index=output_index,
                timestamp=output_index / self.fps,
                image=image,
                path=str(path),
            )
            output_index += 1

    def _iter_image(self) -> Iterator[Frame]:
        path = Path(str(self.source))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {path}")
        self.frame_count = 1
        self.fps = self.default_fps
        yield Frame(index=0, timestamp=0.0, image=image, path=str(path))

    def _iter_capture(self) -> Iterator[Frame]:
        if self.kind == "webcam":
            yield from self._iter_cv_capture(int(self.source), allow_ffmpeg_fallback=False)
            return
        yield from self._iter_cv_capture(str(self.source), allow_ffmpeg_fallback=True)

    def _iter_cv_capture(self, capture_source: str | int, allow_ffmpeg_fallback: bool) -> Iterator[Frame]:
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            cap.release()
            if allow_ffmpeg_fallback:
                LOGGER.warning("OpenCV could not open %s; falling back to ffmpeg", self.source)
                yield from self._iter_video_ffmpeg(str(self.source))
                return
            raise IOError(f"Failed to open video/webcam source: {self.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps and fps > 1e-3 else self.default_fps
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = count if count > 0 else None
        original_index = 0
        output_index = 0
        yielded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if original_index % self.stride == 0:
                    yielded += 1
                    yield Frame(
                        index=output_index,
                        timestamp=original_index / self.fps,
                        image=frame,
                        path=str(self.source),
                    )
                    output_index += 1
                original_index += 1
        finally:
            cap.release()
        if yielded == 0 and allow_ffmpeg_fallback:
            LOGGER.warning("OpenCV opened %s but decoded no frames; falling back to ffmpeg", self.source)
            yield from self._iter_video_ffmpeg(str(self.source))

    def _iter_video_ffmpeg(self, video_path: str) -> Iterator[Frame]:
        if shutil.which("ffmpeg") is None:
            raise IOError(
                f"Failed to decode video with OpenCV and ffmpeg is not installed: {video_path}"
            )
        width, height, fps, frame_count = self._probe_video(video_path)
        self.fps = fps
        self.frame_count = frame_count
        frame_bytes = width * height * 3
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None:
            raise IOError(f"Failed to start ffmpeg decoder for: {video_path}")
        original_index = 0
        output_index = 0
        completed = False
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) == 0:
                    completed = True
                    break
                if len(raw) != frame_bytes:
                    raise IOError(f"Short ffmpeg frame while decoding: {video_path}")
                if original_index % self.stride == 0:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                    yield Frame(
                        index=output_index,
                        timestamp=original_index / self.fps,
                        image=frame,
                        path=video_path,
                    )
                    output_index += 1
                original_index += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if not completed and process.poll() is None:
                process.terminate()
            return_code = process.wait()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            if process.stderr is not None:
                process.stderr.close()
            if completed and return_code not in {0, 255}:
                raise IOError(f"ffmpeg failed while decoding {video_path}: {stderr.strip()}")

    def _probe_video(self, video_path: str) -> tuple[int, int, float, int | None]:
        if shutil.which("ffprobe") is None:
            raise IOError(f"ffprobe is required for ffmpeg video fallback: {video_path}")
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            video_path,
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(completed.stdout)
        streams = data.get("streams") or []
        if not streams:
            raise IOError(f"No video stream found: {video_path}")
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise IOError(f"Invalid video dimensions from ffprobe: {video_path}")
        fps = _parse_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"))
        if fps <= 1e-3:
            fps = self.default_fps
        frame_count: int | None = None
        nb_frames = stream.get("nb_frames")
        if nb_frames not in {None, "N/A"}:
            frame_count = int(nb_frames)
        elif stream.get("duration"):
            frame_count = int(round(float(stream["duration"]) * fps))
        return width, height, fps, frame_count


def _parse_rate(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    den = float(denominator)
    if abs(den) < 1e-9:
        return 0.0
    return float(numerator) / den
