from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from realtime_safety.config import ReconstructionConfig
from realtime_safety.pipeline.mast3r_slam_adapter import (
    Mast3rSlamAdapter,
    _camera_to_internal,
    _read_message,
    _write_message,
)


def test_mast3r_coordinates_are_converted_to_internal_z_up_axes() -> None:
    camera = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0]], np.float32)

    internal = _camera_to_internal(camera)

    assert np.array_equal(
        internal,
        np.array([[1.0, 3.0, -2.0], [-4.0, 6.0, 5.0]], np.float32),
    )


def test_mast3r_binary_protocol_round_trips_numpy_without_disk() -> None:
    stream = io.BytesIO()
    message = {"command": "infer", "rgb": np.arange(24, dtype=np.uint8).reshape(2, 4, 3)}

    _write_message(stream, message)
    stream.seek(0)
    restored = _read_message(stream)

    assert restored["command"] == "infer"
    assert np.array_equal(restored["rgb"], message["rgb"])


def test_mast3r_preflight_reports_isolated_setup_command(tmp_path: Path) -> None:
    config = ReconstructionConfig(
        mast3r_slam_path=str(tmp_path / "missing-slam"),
        mast3r_slam_python=str(tmp_path / "missing-venv" / "bin" / "python"),
    )
    adapter = Mast3rSlamAdapter(config, "cuda")

    with pytest.raises(FileNotFoundError, match="setup_mast3r_slam.sh"):
        adapter.preflight()


def test_mast3r_preflight_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    root = tmp_path / "slam"
    (root / "mast3r_slam").mkdir(parents=True)
    (root / "mast3r_slam" / "frame.py").touch()
    (root / "config").mkdir()
    (root / "config" / "base.yaml").touch()
    (root / "checkpoints").mkdir()
    checkpoint = (
        root
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )
    checkpoint.touch()
    system_python = tmp_path / "system-python"
    system_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)
    config = ReconstructionConfig(
        mast3r_slam_path=str(root),
        mast3r_slam_python=str(venv_python),
        mast3r_slam_loop_closure=False,
    )

    _, actual_python, *_ = Mast3rSlamAdapter(config, "cuda").preflight()

    assert actual_python == venv_python.absolute()
    assert actual_python.is_symlink()
