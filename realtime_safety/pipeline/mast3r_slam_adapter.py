from __future__ import annotations

import logging
import os
import pickle
import select
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np

from realtime_safety.config import ReconstructionConfig
from realtime_safety.types import FramePacket, PointCloudFrame

LOGGER = logging.getLogger(__name__)
_HEADER = struct.Struct("<Q")


class Mast3rSlamAdapter:
    """Isolated live adapter for the official MASt3R-SLAM implementation.

    MASt3R-SLAM and St4RTrack both install top-level ``dust3r`` packages with
    incompatible contents. The SLAM runtime therefore lives in a dedicated
    interpreter and exchanges bounded, in-memory messages with this process.
    No frame is written to disk.
    """

    def __init__(self, config: ReconstructionConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.process: subprocess.Popen[bytes] | None = None
        self.last_gpu_ms = 0.0
        self._project_root = Path(__file__).resolve().parents[2]

    @property
    def available(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def preflight(self) -> tuple[Path, Path, Path, Path, Path]:
        root = Path(
            self.config.mast3r_slam_path or "third_party/MASt3R-SLAM"
        ).expanduser().resolve()
        python = Path(
            self.config.mast3r_slam_python
            or self._project_root / ".venv-mast3r-slam" / "bin" / "python"
        ).expanduser()
        # Do not resolve this symlink: a venv's ``bin/python`` commonly points
        # at the system executable, and replacing argv[0] with that target
        # bypasses the venv's isolated site-packages.
        if not python.is_absolute():
            python = Path.cwd() / python
        python = python.absolute()
        slam_config = Path(
            self.config.mast3r_slam_config or root / "config" / "base.yaml"
        ).expanduser().resolve()
        checkpoint = Path(
            self.config.mast3r_slam_checkpoint
            or root
            / "checkpoints"
            / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        ).expanduser().resolve()
        retrieval = Path(
            self.config.mast3r_slam_retrieval_checkpoint
            or root
            / "checkpoints"
            / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        ).expanduser().resolve()
        retrieval_codebook = retrieval.with_name(
            retrieval.name.replace("_trainingfree.pth", "_codebook.pkl")
        )
        required = {
            "MASt3R-SLAM code": root / "mast3r_slam" / "frame.py",
            "MASt3R-SLAM Python": python,
            "MASt3R-SLAM config": slam_config,
            "MASt3R checkpoint": checkpoint,
        }
        if self.config.mast3r_slam_loop_closure:
            required["MASt3R retrieval checkpoint"] = retrieval
            required["MASt3R retrieval codebook"] = retrieval_codebook
        missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MASt3R-SLAM is not installed completely. Run "
                "`bash scripts/setup_mast3r_slam.sh`. Missing "
                + "; ".join(missing)
            )
        if self.config.mast3r_slam_image_size not in {224, 512}:
            raise ValueError("mast3r_slam_image_size must be 224 or 512")
        return root, python, slam_config, checkpoint, retrieval

    def load(self) -> None:
        if self.available:
            return
        root, python, slam_config, checkpoint, retrieval = self.preflight()
        worker = Path(__file__).with_name("mast3r_slam_worker.py")
        command = [
            str(python),
            "-u",
            str(worker),
            "--root",
            str(root),
            "--config",
            str(slam_config),
            "--checkpoint",
            str(checkpoint),
            "--device",
            self.device,
            "--image-size",
            str(self.config.mast3r_slam_image_size),
            "--confidence-threshold",
            str(self.config.mast3r_slam_confidence_threshold),
            "--max-points",
            str(self.config.max_points),
            "--voxel-size",
            str(self.config.voxel_size),
        ]
        if self.config.mast3r_slam_loop_closure:
            command.extend(["--retrieval-checkpoint", str(retrieval)])
        else:
            command.append("--no-loop-closure")
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        self.process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Upstream progress/errors remain visible without being able to
            # fill a PIPE and deadlock the binary protocol.
            stderr=None,
            env=environment,
        )
        try:
            ready = self._receive(self.config.mast3r_slam_startup_timeout_s)
            if not ready.get("ok") or ready.get("event") != "ready":
                raise RuntimeError(str(ready.get("error", "worker did not become ready")))
        except Exception:
            self.close()
            raise
        LOGGER.info("Loaded isolated MASt3R-SLAM worker from %s using %s", root, python)

    def warmup(self) -> None:
        if not self.available:
            raise RuntimeError("MASt3R-SLAM worker has not been loaded")

    def infer(self, frame: FramePacket) -> PointCloudFrame:
        if not self.available:
            raise RuntimeError("MASt3R-SLAM worker has not been loaded")
        start = time.perf_counter()
        self._send(
            {
                "command": "infer",
                "frame_index": int(frame.frame_index),
                "timestamp": float(frame.source_timestamp),
                "rgb": np.ascontiguousarray(frame.rgb, dtype=np.uint8),
                "max_points": int(self.config.max_points),
                "voxel_size": float(self.config.voxel_size),
            }
        )
        response = self._receive(None)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "MASt3R-SLAM inference failed")))
        pointmap = _camera_to_internal(np.asarray(response["pointmap"], dtype=np.float32))
        points = _camera_to_internal(np.asarray(response["points"], dtype=np.float32))
        colors = np.asarray(response["colors"], dtype=np.uint8).reshape(-1, 3)
        confidence = np.asarray(response["confidence"], dtype=np.float32).reshape(-1)
        dense_confidence = np.asarray(
            response["dense_confidence"], dtype=np.float32
        )
        inference_ms = float(response.get("inference_ms", (time.perf_counter() - start) * 1000.0))
        self.last_gpu_ms = float(response.get("gpu_ms", 0.0))
        return PointCloudFrame(
            points=points,
            colors=colors,
            confidence=confidence,
            pointmap=pointmap,
            frame_index=int(response.get("frame_index", frame.frame_index)),
            timestamp=float(response.get("timestamp", frame.source_timestamp)),
            anchor_frame_index=int(response.get("anchor_frame_index", frame.frame_index)),
            inference_ms=inference_ms,
            valid=len(points) > 0,
            source="mast3r_slam",
            dense_confidence=dense_confidence,
        )

    def reset(self) -> None:
        if not self.available:
            return
        try:
            self._send({"command": "reset"})
            response = self._receive(
                float(self.config.mast3r_slam_startup_timeout_s)
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "reset failed")))
        except Exception as exc:
            LOGGER.warning("Could not reset MASt3R-SLAM worker; restarting it: %s", exc)
            self.close()

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    _write_message(process.stdin, {"command": "close"})
                process.wait(timeout=5.0)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def _send(self, message: object) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("MASt3R-SLAM worker is not running")
        _write_message(process.stdin, message)

    def _receive(self, timeout: float | None) -> dict:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("MASt3R-SLAM worker is not running")
        if timeout is not None:
            readable, _, _ = select.select([process.stdout.fileno()], [], [], timeout)
            if not readable:
                raise TimeoutError(
                    f"MASt3R-SLAM worker did not respond within {timeout:.1f} seconds"
                )
        try:
            response = _read_message(process.stdout)
        except EOFError as exc:
            code = process.poll()
            raise RuntimeError(f"MASt3R-SLAM worker exited unexpectedly (code={code})") from exc
        if not isinstance(response, dict):
            raise RuntimeError("MASt3R-SLAM worker returned an invalid response")
        return response


def _write_message(stream: BinaryIO, message: object) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_message(stream: BinaryIO) -> object:
    header = _read_exact(stream, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size > 512 * 1024 * 1024:
        raise RuntimeError(f"Refusing oversized MASt3R-SLAM message: {size} bytes")
    return pickle.loads(_read_exact(stream, size))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _camera_to_internal(points: np.ndarray) -> np.ndarray:
    """MASt3R x-right/y-down/z-forward -> app x-right/y-forward/z-up."""

    values = np.asarray(points, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError("MASt3R-SLAM points must end in XYZ")
    return np.stack((values[..., 0], values[..., 2], -values[..., 1]), axis=-1)
