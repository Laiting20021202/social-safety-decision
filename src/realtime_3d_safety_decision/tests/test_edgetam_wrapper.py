from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from realtime_safety.edgetam_tracker.edgetam_wrapper import (
    EdgeTAMConfig,
    EdgeTAMConfigurationError,
    EdgeTAMInferenceError,
    EdgeTAMLoadError,
    EdgeTAMWrapper,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_EDGETAM_COMMIT,
)
from realtime_safety.edgetam_tracker.models import ProjectionPrompt


class _FakePredictor:
    def __init__(
        self,
        *,
        fail_box_points: set[int] | None = None,
        fail_propagation: bool = False,
        fail_multi_object_propagation: bool = False,
        first_propagation_started: threading.Event | None = None,
        release_first_propagation: threading.Event | None = None,
        empty_modes: set[str] | None = None,
    ) -> None:
        self.fail_box_points = fail_box_points or set()
        self.fail_propagation = fail_propagation
        self.fail_multi_object_propagation = (
            fail_multi_object_propagation
        )
        self.first_propagation_started = first_propagation_started
        self.release_first_propagation = release_first_propagation
        self.empty_modes = empty_modes or set()
        self.init_frame_counts: list[int] = []
        self.prompt_calls: list[tuple[int, str, int]] = []
        self.propagated_object_sets: list[tuple[int, ...]] = []
        self._propagation_count = 0

    def init_state(self, video_path: str, **_kwargs):
        paths = sorted(Path(video_path).glob("*.jpg"))
        assert paths
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
        assert all(image is not None for image in images)
        self.init_frame_counts.append(len(paths))
        return {
            "num_frames": len(paths),
            "height": images[0].shape[0],
            "width": images[0].shape[1],
            "objects": [],
            "modes": {},
        }

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        box=None,
        **_kwargs,
    ):
        mode = "box_points" if points is not None and len(points) else "box"
        self.prompt_calls.append((int(obj_id), mode, int(frame_idx)))
        if mode == "box_points" and int(obj_id) in self.fail_box_points:
            raise RuntimeError(f"synthetic {mode} rejection for {obj_id}")
        if int(obj_id) not in inference_state["objects"]:
            inference_state["objects"].append(int(obj_id))
        inference_state["modes"][int(obj_id)] = mode
        return self._same_frame_output(inference_state, frame_idx)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        assert np.asarray(mask).ndim == 2
        self.prompt_calls.append((int(obj_id), "mask", int(frame_idx)))
        if int(obj_id) not in inference_state["objects"]:
            inference_state["objects"].append(int(obj_id))
        inference_state["modes"][int(obj_id)] = "mask"
        return self._same_frame_output(inference_state, frame_idx)

    def propagate_in_video(self, inference_state):
        if self.fail_propagation:
            raise RuntimeError("synthetic propagation failure")
        if (
            self.fail_multi_object_propagation
            and len(inference_state["objects"]) > 1
        ):
            raise RuntimeError("synthetic non-contiguous multi-object batch")
        self._propagation_count += 1
        if self._propagation_count == 1 and self.first_propagation_started is not None:
            self.first_propagation_started.set()
            assert self.release_first_propagation is not None
            assert self.release_first_propagation.wait(3.0)
        object_ids = tuple(inference_state["objects"])
        self.propagated_object_sets.append(object_ids)
        for frame_idx in range(inference_state["num_frames"]):
            yield (
                frame_idx,
                list(object_ids),
                self._state_logits(inference_state, object_ids),
            )

    def _same_frame_output(self, state, frame_idx):
        object_ids = list(state["objects"])
        return (
            frame_idx,
            object_ids,
            self._state_logits(state, object_ids),
        )

    def _state_logits(self, state, object_ids) -> np.ndarray:
        logits = self._logits(
            len(object_ids), state["height"], state["width"]
        )
        for index, object_id in enumerate(object_ids):
            if state["modes"].get(int(object_id)) in self.empty_modes:
                logits[index] = -2.0
        return logits

    @staticmethod
    def _logits(count: int, height: int, width: int) -> np.ndarray:
        logits = np.full((count, 1, height, width), -2.0, dtype=np.float32)
        logits[:, :, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = 2.0
        return logits


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "edgetam-test.pt"
    path.write_bytes(b"fake checkpoint for injected official predictor")
    return path


def _config(tmp_path: Path, **overrides) -> EdgeTAMConfig:
    values = {
        "repository_path": None,
        "checkpoint_path": _checkpoint(tmp_path),
        "device": "cpu",
        "precision": "auto",
        "window_size": 3,
        "expected_source_commit": None,
        "expected_checkpoint_sha256": None,
    }
    values.update(overrides)
    return EdgeTAMConfig(**values)


def _prompt(
    track_id: int,
    frame_index: int,
    *,
    with_mask: bool = True,
    re_prompt: bool = False,
) -> ProjectionPrompt:
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 12:36] = True
    return ProjectionPrompt(
        track_id=track_id,
        frame_index=frame_index,
        box_xyxy=np.array([10, 6, 38, 27], dtype=np.float32),
        positive_points=np.array([[20, 16], [28, 17]], dtype=np.float32),
        negative_points=np.array([[4, 4]], dtype=np.float32),
        projection_mask=mask if with_mask else None,
        re_prompt=re_prompt,
    )


def _rgb(value: int) -> np.ndarray:
    return np.full((32, 48, 3), value, dtype=np.uint8)


def test_prompt_fallback_rebuilds_before_propagation_and_thresholds_logits(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor(fail_box_points={1, 2})
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(80),
            10,
            1.0,
            [_prompt(1, 10, with_mask=True), _prompt(2, 10, with_mask=False)],
            active_track_ids=[1, 2],
        )
    finally:
        wrapper.close()

    assert result.ok
    assert result.prompt_modes == {1: "mask", 2: "box"}
    assert set(result.masks) == {1, 2}
    assert all(mask.dtype == np.bool_ and mask.any() for mask in result.masks.values())
    assert predictor.propagated_object_sets[-1] == (1, 2)
    assert predictor.prompt_calls[-2:] == [(1, "mask", 0), (2, "box", 0)]


def test_empty_success_mask_retries_next_public_prompt_mode(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor(empty_modes={"box_points"})
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(80),
            10,
            1.0,
            [_prompt(1, 10, with_mask=True)],
            active_track_ids=[1],
        )
    finally:
        wrapper.close()

    assert result.ok
    assert result.masks[1].any()
    assert result.prompt_modes == {1: "mask+semantic_retry"}
    assert predictor.prompt_calls[-2:] == [
        (1, "box_points", 0),
        (1, "mask", 0),
    ]


def test_all_semantically_empty_prompt_modes_are_explicit_failure(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor(
        empty_modes={"box_points", "mask", "box"}
    )
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(80),
            10,
            1.0,
            [_prompt(1, 10, with_mask=True)],
            active_track_ids=[1],
        )
    finally:
        wrapper.close()

    assert not result.ok
    assert result.masks == {}
    assert "no usable mask" in result.error


def test_object_addition_and_deletion_force_rebuild_with_bounded_window(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor()
    wrapper = EdgeTAMWrapper(
        _config(tmp_path, window_size=2),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        first = wrapper.infer(
            _rgb(40), 0, 0.0, [_prompt(1, 0)], active_track_ids=[1]
        )
        added = wrapper.infer(
            _rgb(60),
            1,
            0.1,
            [_prompt(1, 1), _prompt(2, 1)],
            active_track_ids=[1, 2],
        )
        deleted = wrapper.infer(
            _rgb(80), 2, 0.2, [_prompt(2, 2)], active_track_ids=[2]
        )
    finally:
        wrapper.close()

    assert first.rebuild_reason == "initial"
    assert added.rebuild_reason == "object_set_changed"
    assert deleted.rebuild_reason == "object_set_changed"
    assert set(added.masks) == {1, 2}
    assert set(deleted.masks) == {2}
    assert max(predictor.init_frame_counts) <= 2
    assert predictor.init_frame_counts[-1] == 2
    assert predictor.propagated_object_sets[-2:] == [(1, 2), (2,)]


def test_missing_confirmed_prompt_is_an_explicit_failed_result(tmp_path: Path) -> None:
    predictor = _FakePredictor()
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(100),
            4,
            0.4,
            [],
            active_track_ids=[99],
        )
    finally:
        wrapper.close()

    assert not result.ok
    assert result.masks == {}
    assert "track 99 has no prompt" in result.error
    with pytest.raises(EdgeTAMInferenceError):
        result.raise_for_error()


def test_propagation_exception_never_becomes_empty_success(tmp_path: Path) -> None:
    predictor = _FakePredictor(fail_propagation=True)
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(100),
            5,
            0.5,
            [_prompt(5, 5)],
            active_track_ids=[5],
        )
    finally:
        wrapper.close()

    assert result.ok is False
    assert result.masks == {}
    assert "synthetic propagation failure" in result.error
    assert result.exception_type == "EdgeTAMInferenceError"


def test_multi_object_propagation_uses_explicit_independent_state_fallback(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor(fail_multi_object_propagation=True)
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        result = wrapper.infer(
            _rgb(100),
            5,
            0.5,
            [_prompt(5, 5), _prompt(8, 5)],
            active_track_ids=[5, 8],
        )
    finally:
        wrapper.close()

    assert result.ok
    assert set(result.masks) == {5, 8}
    assert result.prompt_modes == {
        5: "box_points+independent_state",
        8: "box_points+independent_state",
    }
    assert predictor.propagated_object_sets[-2:] == [(5,), (8,)]


def test_latest_only_queue_drops_stale_pending_job(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    predictor = _FakePredictor(
        first_propagation_started=started,
        release_first_propagation=release,
    )
    wrapper = EdgeTAMWrapper(
        _config(tmp_path, window_size=2),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        wrapper.submit(_rgb(10), 0, 0.0, [_prompt(1, 0)])
        assert started.wait(3.0)
        wrapper.submit(_rgb(20), 1, 0.1, [_prompt(1, 1)])
        newest_sequence = wrapper.submit(_rgb(30), 2, 0.2, [_prompt(1, 2)])
        release.set()
        newest = wrapper.wait_for_result(newest_sequence, timeout=5.0)
    finally:
        release.set()
        wrapper.close()

    assert newest.frame_index == 2
    assert newest.sequence == newest_sequence
    assert newest.dropped_jobs >= 1
    assert wrapper.dropped_jobs >= 1


def test_reset_stream_discards_an_inflight_old_generation_result(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    predictor = _FakePredictor(
        first_propagation_started=started,
        release_first_propagation=release,
    )
    wrapper = EdgeTAMWrapper(
        _config(tmp_path),
        predictor_builder=lambda *_args, **_kwargs: predictor,
    )
    try:
        wrapper.load()
        wrapper.submit(_rgb(10), 50, 5.0, [_prompt(1, 50)])
        assert started.wait(3.0)
        wrapper.reset_stream()
        assert wrapper.latest_result is None
        release.set()
        wrapper._jobs.join()
        assert wrapper.latest_result is None

        sequence = wrapper.submit(
            _rgb(20),
            0,
            0.0,
            [_prompt(7, 0)],
            active_track_ids=[7],
        )
        current = wrapper.wait_for_result(sequence, timeout=5.0)
    finally:
        release.set()
        wrapper.close()

    assert current.ok
    assert current.stream_generation > 0
    assert current.frame_index == 0
    assert set(current.masks) == {7}


def test_load_and_precision_validation_are_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    wrapper = EdgeTAMWrapper(
        EdgeTAMConfig(
            repository_path=None,
            checkpoint_path=missing,
            device="cpu",
            precision="fp32",
            expected_checkpoint_sha256=None,
        ),
        predictor_builder=lambda *_args, **_kwargs: _FakePredictor(),
    )
    with pytest.raises(EdgeTAMLoadError, match="checkpoint not found"):
        wrapper.load()

    half_wrapper = EdgeTAMWrapper(
        _config(tmp_path, device="cpu", precision="fp16"),
        predictor_builder=lambda *_args, **_kwargs: _FakePredictor(),
    )
    with pytest.raises(EdgeTAMConfigurationError, match="only enabled for CUDA"):
        half_wrapper.load()

    with pytest.raises(EdgeTAMConfigurationError, match="Unsupported"):
        _config(tmp_path, precision="int8")
    with pytest.raises(
        EdgeTAMConfigurationError,
        match="expected_source_commit",
    ):
        _config(tmp_path, expected_source_commit="not-a-commit")


def test_setup_and_checkpoint_scripts_pin_verified_official_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    setup = root / "scripts" / "setup_edgetam.sh"
    download = root / "scripts" / "download_edgetam_checkpoint.sh"
    subprocess.run(["bash", "-n", str(setup)], check=True)
    subprocess.run(["bash", "-n", str(download)], check=True)
    assert OFFICIAL_EDGETAM_COMMIT in setup.read_text(encoding="utf-8")
    assert OFFICIAL_CHECKPOINT_SHA256 in download.read_text(encoding="utf-8")
    assert "sys.prefix == sys.base_prefix" in setup.read_text(encoding="utf-8")
