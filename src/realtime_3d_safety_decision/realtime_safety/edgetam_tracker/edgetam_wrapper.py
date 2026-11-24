from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from realtime_safety.edgetam_tracker.models import ProjectionPrompt

LOGGER = logging.getLogger(__name__)

OFFICIAL_EDGETAM_COMMIT = "7711e012a30a2402c4eaab637bdb00a521302c91"
OFFICIAL_CHECKPOINT_SHA256 = (
    "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
)
SUPPORTED_PRECISIONS = frozenset({"auto", "bf16", "fp16", "fp32"})


class EdgeTAMError(RuntimeError):
    """Base class for explicit EdgeTAM adapter failures."""


class EdgeTAMConfigurationError(EdgeTAMError):
    """The wrapper configuration cannot be used safely."""


class EdgeTAMLoadError(EdgeTAMError):
    """The official model or checkpoint could not be loaded."""


class EdgeTAMInferenceError(EdgeTAMError):
    """A window could not produce a complete set of object masks."""


@dataclass(slots=True)
class EdgeTAMConfig:
    """Runtime settings for the pinned official EdgeTAM video predictor.

    EdgeTAM's public video API accepts a finite MP4 or numeric JPEG directory,
    not a live frame iterator. This wrapper therefore materializes a bounded
    rolling JPEG window and creates a fresh upstream inference state for each
    processed latest frame.
    """

    repository_path: str | Path | None
    checkpoint_path: str | Path
    model_config: str = "configs/edgetam.yaml"
    device: str = "cuda"
    precision: str = "auto"
    window_size: int = 8
    jpeg_quality: int = 95
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    clear_memory_on_reset: bool = True
    expected_source_commit: str | None = OFFICIAL_EDGETAM_COMMIT
    expected_checkpoint_sha256: str | None = OFFICIAL_CHECKPOINT_SHA256
    temporary_directory: str | Path | None = None

    def __post_init__(self) -> None:
        self.precision = str(self.precision).lower()
        if self.precision not in SUPPORTED_PRECISIONS:
            allowed = ", ".join(sorted(SUPPORTED_PRECISIONS))
            raise EdgeTAMConfigurationError(
                f"Unsupported EdgeTAM precision {self.precision!r}; expected one of: {allowed}"
            )
        self.device = str(self.device).lower()
        if not self.device:
            raise EdgeTAMConfigurationError("EdgeTAM device must not be empty")
        if not self.model_config:
            raise EdgeTAMConfigurationError("EdgeTAM model_config must not be empty")
        if self.window_size < 1:
            raise EdgeTAMConfigurationError("EdgeTAM window_size must be at least 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise EdgeTAMConfigurationError(
                "EdgeTAM jpeg_quality must be in the inclusive range [1, 100]"
            )
        if self.expected_checkpoint_sha256 is not None:
            digest = self.expected_checkpoint_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise EdgeTAMConfigurationError(
                    "expected_checkpoint_sha256 must be a 64-character hexadecimal digest"
                )
            self.expected_checkpoint_sha256 = digest
        if self.expected_source_commit is not None:
            commit = self.expected_source_commit.lower()
            if len(commit) != 40 or any(
                char not in "0123456789abcdef" for char in commit
            ):
                raise EdgeTAMConfigurationError(
                    "expected_source_commit must be a 40-character "
                    "hexadecimal Git commit"
                )
            self.expected_source_commit = commit


@dataclass(slots=True)
class EdgeTAMResult:
    """One latest-frame result.

    An empty ``masks`` mapping is only valid when ``ok`` is false. This makes it
    impossible for an inference exception to masquerade as a safe empty scene.
    """

    sequence: int
    stream_generation: int
    frame_index: int
    stamp: float
    ok: bool
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    prompt_modes: dict[int, str] = field(default_factory=dict)
    error: str = ""
    exception_type: str = ""
    rebuild_reason: str = ""
    window_start_frame: int | None = None
    window_end_frame: int | None = None
    window_frame_count: int = 0
    dropped_jobs: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.ok and not self.masks:
            raise ValueError("A successful EdgeTAM result must contain at least one mask")
        if not self.ok and not self.error:
            raise ValueError("A failed EdgeTAM result must include an error message")
        self.masks = {
            int(track_id): np.asarray(mask, dtype=bool)
            for track_id, mask in self.masks.items()
        }

    def raise_for_error(self) -> None:
        if not self.ok:
            raise EdgeTAMInferenceError(self.error)


@dataclass(slots=True)
class _WindowFrame:
    frame_index: int
    stamp: float
    rgb: np.ndarray
    prompts: dict[int, ProjectionPrompt]


@dataclass(slots=True)
class _InferenceJob:
    sequence: int
    stream_generation: int
    frame_index: int
    stamp: float
    window: tuple[_WindowFrame, ...]
    active_track_ids: tuple[int, ...]
    rebuild_reason: str


@dataclass(slots=True)
class _PromptFailure(Exception):
    track_id: int
    mode: str
    cause: Exception


_STOP = object()


class EdgeTAMWrapper:
    """Single-worker, latest-only adapter around official EdgeTAM APIs.

    The optional ``predictor_builder`` argument exists for dependency injection
    in unit tests. Production code should omit it so ``load`` dynamically
    imports ``sam2.build_sam.build_sam2_video_predictor`` from the configured
    official repository.
    """

    def __init__(
        self,
        config: EdgeTAMConfig,
        *,
        predictor_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._injected_builder = predictor_builder
        self._predictor: Any | None = None
        self._torch: Any | None = None
        self._resolved_device = ""
        self._resolved_precision = ""
        self._autocast_dtype: Any | None = None

        self._frames: deque[_WindowFrame] = deque(maxlen=config.window_size)
        self._last_submitted_ids: frozenset[int] | None = None
        self._last_submitted_frame: int | None = None
        self._image_shape: tuple[int, int] | None = None
        self._sequence = 0
        self._stream_generation = 0
        self._dropped_jobs = 0
        self._submit_lock = threading.Lock()

        self._jobs: queue.Queue[_InferenceJob | object] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._result_condition = threading.Condition()
        self._latest_result: EdgeTAMResult | None = None

    @property
    def available(self) -> bool:
        return self._predictor is not None

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    @property
    def resolved_precision(self) -> str:
        return self._resolved_precision

    @property
    def dropped_jobs(self) -> int:
        with self._submit_lock:
            return self._dropped_jobs

    @property
    def latest_result(self) -> EdgeTAMResult | None:
        with self._result_condition:
            return self._latest_result

    def load(self, *, start_worker: bool = True) -> None:
        """Load the pinned official predictor or raise an actionable error."""

        if self.available:
            if start_worker:
                self.start()
            return

        checkpoint = Path(self.config.checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise EdgeTAMLoadError(
                f"EdgeTAM checkpoint not found at {checkpoint}. "
                "Run scripts/download_edgetam_checkpoint.sh or set checkpoint_path."
            )
        if checkpoint.stat().st_size <= 0:
            raise EdgeTAMLoadError(f"EdgeTAM checkpoint is empty: {checkpoint}")
        if self.config.expected_checkpoint_sha256 is not None:
            actual_digest = _sha256(checkpoint)
            if actual_digest != self.config.expected_checkpoint_sha256:
                raise EdgeTAMLoadError(
                    "EdgeTAM checkpoint checksum mismatch for "
                    f"{checkpoint}: expected {self.config.expected_checkpoint_sha256}, "
                    f"got {actual_digest}. Re-download the official checkpoint."
                )

        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise EdgeTAMLoadError(
                "PyTorch is required for EdgeTAM. Install it inside the project virtual environment."
            ) from exc

        device, precision, autocast_dtype = self._resolve_runtime(torch)
        if device.startswith("cuda"):
            # EdgeTAM has fixed image shapes in this live deployment. Let
            # cuDNN cache the fastest kernels and use TensorFloat-32 for the
            # few FP32 matrix operations outside autocast.
            matmul_backend = torch.backends.cuda.matmul
            convolution_backend = getattr(
                torch.backends.cudnn, "conv", None
            )
            if hasattr(matmul_backend, "fp32_precision"):
                matmul_backend.fp32_precision = "tf32"
            else:
                matmul_backend.allow_tf32 = True
            if (
                convolution_backend is not None
                and hasattr(convolution_backend, "fp32_precision")
            ):
                convolution_backend.fp32_precision = "tf32"
            else:
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            set_matmul_precision = getattr(
                torch, "set_float32_matmul_precision", None
            )
            if callable(set_matmul_precision):
                set_matmul_precision("high")
        builder = self._injected_builder or self._import_official_builder()
        try:
            with torch.inference_mode():
                predictor = builder(
                    self.config.model_config,
                    str(checkpoint),
                    device=device,
                )
        except Exception as exc:
            raise EdgeTAMLoadError(
                "Failed to build the official EdgeTAM video predictor "
                f"from config {self.config.model_config!r} and checkpoint {checkpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if predictor is None:
            raise EdgeTAMLoadError(
                "Official build_sam2_video_predictor returned None instead of a predictor"
            )
        for method_name in (
            "init_state",
            "add_new_points_or_box",
            "add_new_mask",
            "propagate_in_video",
        ):
            if not callable(getattr(predictor, method_name, None)):
                raise EdgeTAMLoadError(
                    f"Loaded EdgeTAM predictor is missing required public method {method_name}"
                )

        self._torch = torch
        self._predictor = predictor
        self._resolved_device = device
        self._resolved_precision = precision
        self._autocast_dtype = autocast_dtype
        LOGGER.info(
            "Loaded EdgeTAM commit %s on %s with %s precision",
            OFFICIAL_EDGETAM_COMMIT,
            device,
            precision,
        )
        if start_worker:
            self.start()

    def start(self) -> None:
        if not self.available:
            raise EdgeTAMLoadError("Call EdgeTAMWrapper.load() before start()")
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="edgetam-latest-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        rgb: np.ndarray,
        frame_index: int,
        stamp: float,
        prompts: Sequence[ProjectionPrompt],
        *,
        active_track_ids: Iterable[int] | None = None,
    ) -> int:
        """Queue the newest frame and replace any older pending inference job.

        ``active_track_ids`` must contain every confirmed object that EdgeTAM is
        expected to propagate. If omitted it is derived from ``prompts``. An
        active ID without any usable prompt in the rolling window makes the
        result fail explicitly.
        """

        if not self.available:
            raise EdgeTAMLoadError("EdgeTAM model has not been loaded")
        self.start()
        frame = _validate_rgb(rgb)
        prompt_map = _prompt_map(prompts)
        active_ids = tuple(
            sorted(
                {
                    int(track_id)
                    for track_id in (
                        prompt_map if active_track_ids is None else active_track_ids
                    )
                }
            )
        )
        if not active_ids:
            raise EdgeTAMConfigurationError(
                "At least one confirmed active track is required for EdgeTAM inference"
            )

        with self._submit_lock:
            reason = "rolling_window"
            image_shape = frame.shape[:2]
            if self._last_submitted_ids is None:
                reason = "initial"
            elif frozenset(active_ids) != self._last_submitted_ids:
                reason = "object_set_changed"
            if self._image_shape is not None and image_shape != self._image_shape:
                self._frames.clear()
                reason = "image_shape_changed"
            if (
                self._last_submitted_frame is not None
                and int(frame_index) <= self._last_submitted_frame
            ):
                self._frames.clear()
                reason = "stream_reset"

            window_frame = _WindowFrame(
                frame_index=int(frame_index),
                stamp=float(stamp),
                rgb=frame.copy(),
                prompts={track_id: _copy_prompt(prompt) for track_id, prompt in prompt_map.items()},
            )
            self._frames.append(window_frame)
            self._sequence += 1
            sequence = self._sequence
            job = _InferenceJob(
                sequence=sequence,
                stream_generation=self._stream_generation,
                frame_index=int(frame_index),
                stamp=float(stamp),
                window=tuple(self._frames),
                active_track_ids=active_ids,
                rebuild_reason=reason,
            )
            self._last_submitted_ids = frozenset(active_ids)
            self._last_submitted_frame = int(frame_index)
            self._image_shape = image_shape
            self._put_latest_locked(job)
        return sequence

    def infer(
        self,
        rgb: np.ndarray,
        frame_index: int,
        stamp: float,
        prompts: Sequence[ProjectionPrompt],
        *,
        active_track_ids: Iterable[int] | None = None,
        timeout: float = 30.0,
    ) -> EdgeTAMResult:
        """Submit a frame and wait for this or a newer latest-only result."""

        sequence = self.submit(
            rgb,
            frame_index,
            stamp,
            prompts,
            active_track_ids=active_track_ids,
        )
        return self.wait_for_result(sequence, timeout=timeout)

    def wait_for_result(self, sequence: int, *, timeout: float = 30.0) -> EdgeTAMResult:
        deadline = time.monotonic() + timeout
        with self._result_condition:
            while (
                self._latest_result is None
                or self._latest_result.sequence < int(sequence)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for EdgeTAM result sequence {sequence}"
                    )
                self._result_condition.wait(remaining)
            return self._latest_result

    def reset_stream(self) -> None:
        """Forget rolling frames and object membership without unloading the model."""

        with self._submit_lock:
            self._stream_generation += 1
            self._frames.clear()
            self._last_submitted_ids = None
            self._last_submitted_frame = None
            self._image_shape = None
            self._discard_pending_locked()
            with self._result_condition:
                self._latest_result = None
                self._result_condition.notify_all()
        if (
            self.config.clear_memory_on_reset
            and self._torch is not None
            and self._resolved_device.startswith("cuda")
        ):
            self._torch.cuda.empty_cache()

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._submit_lock:
            # Invalidate a job already executing before waiting for it to
            # finish, so shutdown can never expose its late result.
            self._stream_generation += 1
            self._discard_pending_locked()
            try:
                self._jobs.put_nowait(_STOP)
            except queue.Full:
                pass
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout)
            if worker.is_alive():
                LOGGER.warning("EdgeTAM worker did not stop within %.1f seconds", timeout)
        self._worker = None
        self._predictor = None
        self._torch = None
        self.reset_stream()

    def _resolve_runtime(self, torch: Any) -> tuple[str, str, Any | None]:
        requested_device = self.config.device
        if requested_device == "auto":
            if bool(torch.cuda.is_available()):
                device = "cuda"
            elif bool(getattr(torch.backends, "mps", None)) and bool(
                torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        else:
            device = requested_device

        if device.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise EdgeTAMConfigurationError(
                f"EdgeTAM device {device!r} requested, but torch.cuda.is_available() is false"
            )
        if device.startswith("mps"):
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not bool(mps.is_available()):
                raise EdgeTAMConfigurationError(
                    "EdgeTAM MPS device requested, but torch.backends.mps is unavailable"
                )

        precision = self.config.precision
        if precision == "auto":
            if device.startswith("cuda"):
                precision = (
                    "bf16"
                    if bool(torch.cuda.is_bf16_supported())
                    else "fp16"
                )
            else:
                precision = "fp32"

        if precision in {"bf16", "fp16"} and not device.startswith("cuda"):
            raise EdgeTAMConfigurationError(
                f"EdgeTAM {precision} autocast is only enabled for CUDA devices; "
                f"device is {device!r}. Use fp32."
            )
        if precision == "bf16" and not bool(torch.cuda.is_bf16_supported()):
            raise EdgeTAMConfigurationError(
                "EdgeTAM bf16 was requested, but this CUDA device does not support bf16"
            )
        dtype = {
            "bf16": getattr(torch, "bfloat16", None),
            "fp16": getattr(torch, "float16", None),
            "fp32": None,
        }[precision]
        if precision != "fp32" and dtype is None:
            raise EdgeTAMConfigurationError(
                f"Installed PyTorch does not expose the dtype required for {precision}"
            )
        return device, precision, dtype

    def _import_official_builder(self) -> Callable[..., Any]:
        repository_path = self.config.repository_path
        if repository_path is not None:
            root = Path(repository_path).expanduser().resolve()
            build_module = root / "sam2" / "build_sam.py"
            if not build_module.is_file():
                raise EdgeTAMLoadError(
                    f"Official EdgeTAM source not found at {root}; expected {build_module}. "
                    "Run scripts/setup_edgetam.sh or correct repository_path."
                )
            if self.config.expected_source_commit is not None:
                try:
                    completed = subprocess.run(
                        ["git", "-C", str(root), "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise EdgeTAMLoadError(
                        "Could not verify the EdgeTAM source commit at "
                        f"{root}: {type(exc).__name__}: {exc}"
                    ) from exc
                actual_commit = completed.stdout.strip().lower()
                if actual_commit != self.config.expected_source_commit:
                    raise EdgeTAMLoadError(
                        "EdgeTAM source commit mismatch at "
                        f"{root}: expected "
                        f"{self.config.expected_source_commit}, got "
                        f"{actual_commit or '<empty>'}. Run "
                        "scripts/setup_edgetam.sh."
                    )
            loaded_sam2 = sys.modules.get("sam2")
            if loaded_sam2 is not None:
                module_file = getattr(loaded_sam2, "__file__", None)
                if module_file is not None and not _is_relative_to(
                    Path(module_file).resolve(), root
                ):
                    raise EdgeTAMLoadError(
                        "A different 'sam2' package is already imported from "
                        f"{module_file}; refusing to mix it with EdgeTAM at {root}. "
                        "Start a clean process with the pinned EdgeTAM repository first on sys.path."
                    )
            root_string = str(root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            importlib.invalidate_caches()
        try:
            module = importlib.import_module("sam2.build_sam")
        except Exception as exc:
            raise EdgeTAMLoadError(
                "Could not import official sam2.build_sam. "
                "Run scripts/setup_edgetam.sh inside the project virtual environment. "
                f"Import failed with {type(exc).__name__}: {exc}"
            ) from exc
        builder = getattr(module, "build_sam2_video_predictor", None)
        if not callable(builder):
            raise EdgeTAMLoadError(
                "sam2.build_sam does not expose build_sam2_video_predictor; "
                f"expected pinned EdgeTAM commit {OFFICIAL_EDGETAM_COMMIT}"
            )
        return builder

    def _put_latest_locked(self, job: _InferenceJob) -> None:
        try:
            self._jobs.put_nowait(job)
            return
        except queue.Full:
            pass
        try:
            dropped = self._jobs.get_nowait()
            self._jobs.task_done()
            if dropped is not _STOP:
                self._dropped_jobs += 1
        except queue.Empty:
            pass
        self._jobs.put_nowait(job)

    def _discard_pending_locked(self) -> None:
        while True:
            try:
                pending = self._jobs.get_nowait()
                self._jobs.task_done()
                if pending is not _STOP:
                    self._dropped_jobs += 1
            except queue.Empty:
                return

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _InferenceJob)
                result = self._process_job(item)
                with self._submit_lock:
                    if item.stream_generation != self._stream_generation:
                        continue
                    with self._result_condition:
                        if (
                            self._latest_result is None
                            or result.sequence >= self._latest_result.sequence
                        ):
                            self._latest_result = result
                        self._result_condition.notify_all()
            finally:
                self._jobs.task_done()

    def _process_job(self, job: _InferenceJob) -> EdgeTAMResult:
        started = time.perf_counter()
        window_start = job.window[0].frame_index if job.window else None
        window_end = job.window[-1].frame_index if job.window else None
        try:
            masks, prompt_modes = self._infer_window(job)
            return EdgeTAMResult(
                sequence=job.sequence,
                stream_generation=job.stream_generation,
                frame_index=job.frame_index,
                stamp=job.stamp,
                ok=True,
                masks=masks,
                prompt_modes=prompt_modes,
                rebuild_reason=job.rebuild_reason,
                window_start_frame=window_start,
                window_end_frame=window_end,
                window_frame_count=len(job.window),
                dropped_jobs=self.dropped_jobs,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            message = (
                f"EdgeTAM inference failed for frame {job.frame_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            LOGGER.exception(message)
            return EdgeTAMResult(
                sequence=job.sequence,
                stream_generation=job.stream_generation,
                frame_index=job.frame_index,
                stamp=job.stamp,
                ok=False,
                masks={},
                error=message,
                exception_type=type(exc).__name__,
                rebuild_reason=job.rebuild_reason,
                window_start_frame=window_start,
                window_end_frame=window_end,
                window_frame_count=len(job.window),
                dropped_jobs=self.dropped_jobs,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _infer_window(
        self, job: _InferenceJob
    ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        if self._predictor is None or self._torch is None:
            raise EdgeTAMLoadError("EdgeTAM predictor became unavailable")
        if not job.window:
            raise EdgeTAMInferenceError("Rolling JPEG window is empty")
        selected = _select_active_prompts(job.window, job.active_track_ids)
        temp_parent = (
            None
            if self.config.temporary_directory is None
            else str(Path(self.config.temporary_directory).expanduser().resolve())
        )
        with tempfile.TemporaryDirectory(
            prefix="edgetam-window-",
            dir=temp_parent,
        ) as directory:
            _write_jpeg_window(
                job.window,
                Path(directory),
                jpeg_quality=self.config.jpeg_quality,
            )
            with self._inference_context():
                state, prompt_modes = self._initialize_prompted_state(
                    directory,
                    selected,
                    job.active_track_ids,
                )
                latest_local_index = len(job.window) - 1
                try:
                    masks = self._propagate_latest(
                        state,
                        latest_local_index,
                    )
                except EdgeTAMInferenceError as grouped_error:
                    if len(job.active_track_ids) <= 1:
                        raise
                    grouped_message = str(grouped_error)
                    LOGGER.warning(
                        "Grouped official EdgeTAM propagation failed for IDs %s; "
                        "retrying each ID in an independent official predictor "
                        "state: %s",
                        job.active_track_ids,
                        grouped_message,
                    )
                    # The failed grouped state can be large. Release both the
                    # direct reference and the traceback reference before
                    # allocating sequential single-object states.
                    grouped_error.__traceback__ = None
                    del state
                    try:
                        masks, prompt_modes = (
                            self._infer_independent_object_states(
                                directory,
                                selected,
                                job.active_track_ids,
                                latest_local_index,
                            )
                        )
                    except Exception as independent_error:
                        raise EdgeTAMInferenceError(
                            "Grouped official EdgeTAM propagation failed and "
                            "the independent-object compatibility fallback also "
                            f"failed. Grouped error: {grouped_message}; "
                            "independent error: "
                            f"{type(independent_error).__name__}: "
                            f"{independent_error}"
                        ) from independent_error
                masks, prompt_modes = self._retry_semantically_invalid_masks(
                    directory,
                    selected,
                    masks,
                    prompt_modes,
                    latest_local_index,
                )
        missing = set(job.active_track_ids) - set(masks)
        if missing:
            raise EdgeTAMInferenceError(
                f"Official video output omitted active object IDs: {sorted(missing)}"
            )
        return {track_id: masks[track_id] for track_id in job.active_track_ids}, prompt_modes

    def _retry_semantically_invalid_masks(
        self,
        video_directory: str,
        selected: Mapping[int, tuple[int, ProjectionPrompt]],
        masks: Mapping[int, np.ndarray],
        prompt_modes: Mapping[int, str],
        latest_local_index: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """Retry the next public prompt API after an empty/irrelevant mask.

        The official predictor can accept a box+points call without raising
        yet propagate an all-false mask.  API success is not segmentation
        success, and repeatedly rebuilding with the same prompt mode leaves
        the tracker in a permanent empty-mask loop.  Retry only the affected
        object in a fresh official state so other valid objects are retained.
        """

        resolved_masks = {
            int(track_id): np.asarray(mask, dtype=bool)
            for track_id, mask in masks.items()
        }
        resolved_modes = {
            int(track_id): str(mode)
            for track_id, mode in prompt_modes.items()
        }
        failures: dict[int, list[str]] = {}
        for track_id, (prompt_frame, prompt) in selected.items():
            current = resolved_masks.get(track_id)
            issue = _semantic_mask_issue(
                current,
                prompt,
                prompt_frame=prompt_frame,
                latest_frame=latest_local_index,
            )
            if not issue:
                continue
            failures[track_id] = [
                f"{resolved_modes.get(track_id, 'unknown')}: {issue}"
            ]
            strategies = _prompt_strategies(prompt)
            used_mode = resolved_modes.get(track_id, "").split("+", 1)[0]
            try:
                start = strategies.index(used_mode) + 1
            except ValueError:
                start = 0
            recovered = False
            for mode in strategies[start:]:
                try:
                    state = self._predictor.init_state(
                        video_path=video_directory,
                        offload_video_to_cpu=self.config.offload_video_to_cpu,
                        offload_state_to_cpu=self.config.offload_state_to_cpu,
                        async_loading_frames=self.config.async_loading_frames,
                    )
                    self._apply_prompt(
                        state,
                        prompt_frame,
                        track_id,
                        prompt,
                        mode,
                    )
                    candidate_masks = self._propagate_latest(
                        state,
                        latest_local_index,
                    )
                    candidate = candidate_masks.get(track_id)
                    candidate_issue = _semantic_mask_issue(
                        candidate,
                        prompt,
                        prompt_frame=prompt_frame,
                        latest_frame=latest_local_index,
                    )
                except Exception as exc:
                    candidate_issue = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    candidate = None
                if candidate_issue:
                    failures[track_id].append(
                        f"{mode}: {candidate_issue}"
                    )
                    continue
                assert candidate is not None
                resolved_masks[track_id] = np.asarray(
                    candidate, dtype=bool
                )
                resolved_modes[track_id] = f"{mode}+semantic_retry"
                recovered = True
                break
            if not recovered:
                raise EdgeTAMInferenceError(
                    "Official EdgeTAM returned no usable mask for track "
                    f"{track_id} after bounded prompt retries: "
                    + "; ".join(failures[track_id])
                )
        return resolved_masks, resolved_modes

    def _propagate_latest(
        self,
        state: Any,
        latest_local_index: int,
    ) -> dict[int, np.ndarray]:
        latest_output: tuple[Sequence[Any], Any] | None = None
        try:
            iterator = self._predictor.propagate_in_video(state)
            for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                if int(out_frame_idx) == latest_local_index:
                    latest_output = (out_obj_ids, out_mask_logits)
        except Exception as exc:
            raise EdgeTAMInferenceError(
                "Official propagate_in_video failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if latest_output is None:
            raise EdgeTAMInferenceError(
                "Official propagate_in_video produced no output for the "
                "latest rolling-window frame"
            )
        return _public_video_masks(*latest_output)

    def _infer_independent_object_states(
        self,
        video_directory: str,
        selected: Mapping[int, tuple[int, ProjectionPrompt]],
        active_track_ids: Sequence[int],
        latest_local_index: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """Retry a failed grouped call with one official state per object.

        Pinned upstream EdgeTAM can fail in its multi-object batch path with
        newer PyTorch releases when an expanded tensor is viewed as contiguous.
        Keeping one public predictor state per object avoids modifying model
        internals while preserving complete masks and explicit degraded-mode
        reporting through the prompt-mode suffix.
        """

        masks: dict[int, np.ndarray] = {}
        prompt_modes: dict[int, str] = {}
        for track_id in active_track_ids:
            state, modes = self._initialize_prompted_state(
                video_directory,
                selected,
                (track_id,),
            )
            object_masks = self._propagate_latest(
                state,
                latest_local_index,
            )
            if track_id not in object_masks:
                raise EdgeTAMInferenceError(
                    "Independent official EdgeTAM state omitted active object "
                    f"ID {track_id}"
                )
            masks[track_id] = object_masks[track_id]
            prompt_modes[track_id] = (
                f"{modes[track_id]}+independent_state"
            )
        return masks, prompt_modes

    def _initialize_prompted_state(
        self,
        video_directory: str,
        selected: Mapping[int, tuple[int, ProjectionPrompt]],
        active_track_ids: Sequence[int],
    ) -> tuple[Any, dict[int, str]]:
        strategies = {
            track_id: _prompt_strategies(selected[track_id][1])
            for track_id in active_track_ids
        }
        strategy_indices = {track_id: 0 for track_id in active_track_ids}
        errors: dict[int, list[str]] = {track_id: [] for track_id in active_track_ids}
        maximum_attempts = sum(len(items) for items in strategies.values()) + 1

        for _ in range(maximum_attempts):
            try:
                state = self._predictor.init_state(
                    video_path=video_directory,
                    offload_video_to_cpu=self.config.offload_video_to_cpu,
                    offload_state_to_cpu=self.config.offload_state_to_cpu,
                    async_loading_frames=self.config.async_loading_frames,
                )
            except Exception as exc:
                raise EdgeTAMInferenceError(
                    f"Official init_state failed: {type(exc).__name__}: {exc}"
                ) from exc

            used_modes: dict[int, str] = {}
            try:
                for track_id in active_track_ids:
                    local_frame_index, prompt = selected[track_id]
                    mode = strategies[track_id][strategy_indices[track_id]]
                    try:
                        self._apply_prompt(
                            state,
                            local_frame_index,
                            track_id,
                            prompt,
                            mode,
                        )
                    except Exception as exc:
                        raise _PromptFailure(track_id, mode, exc) from exc
                    used_modes[track_id] = mode
                return state, used_modes
            except _PromptFailure as failure:
                errors[failure.track_id].append(
                    f"{failure.mode}: {type(failure.cause).__name__}: {failure.cause}"
                )
                strategy_indices[failure.track_id] += 1
                if strategy_indices[failure.track_id] >= len(
                    strategies[failure.track_id]
                ):
                    attempts = "; ".join(errors[failure.track_id])
                    raise EdgeTAMInferenceError(
                        f"All prompt strategies failed for track {failure.track_id}: {attempts}"
                    ) from failure.cause
                LOGGER.warning(
                    "EdgeTAM prompt %s failed for track %d; rebuilding state and trying %s",
                    failure.mode,
                    failure.track_id,
                    strategies[failure.track_id][strategy_indices[failure.track_id]],
                )
        raise EdgeTAMInferenceError("Exceeded bounded EdgeTAM prompt fallback attempts")

    def _apply_prompt(
        self,
        state: Any,
        frame_index: int,
        track_id: int,
        prompt: ProjectionPrompt,
        mode: str,
    ) -> None:
        if mode == "box_points":
            positive = _finite_points(prompt.positive_points)
            negative = _finite_points(prompt.negative_points)
            points = np.concatenate((positive, negative), axis=0)
            labels = np.concatenate(
                (
                    np.ones(len(positive), dtype=np.int32),
                    np.zeros(len(negative), dtype=np.int32),
                )
            )
            self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=track_id,
                points=points,
                labels=labels,
                clear_old_points=True,
                normalize_coords=True,
                box=np.asarray(prompt.box_xyxy, dtype=np.float32),
            )
            return
        if mode == "mask":
            self._predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=track_id,
                mask=np.asarray(prompt.projection_mask, dtype=bool),
            )
            return
        if mode == "box":
            self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=track_id,
                points=None,
                labels=None,
                clear_old_points=True,
                normalize_coords=True,
                box=np.asarray(prompt.box_xyxy, dtype=np.float32),
            )
            return
        raise EdgeTAMConfigurationError(f"Unknown prompt strategy {mode!r}")

    def _inference_context(self) -> contextlib.AbstractContextManager[Any]:
        assert self._torch is not None
        inference_mode = self._torch.inference_mode()
        if self._resolved_precision == "fp32":
            autocast = contextlib.nullcontext()
        else:
            autocast = self._torch.autocast(
                device_type="cuda",
                dtype=self._autocast_dtype,
            )
        return _CombinedContext(inference_mode, autocast)


class _CombinedContext(contextlib.AbstractContextManager[Any]):
    def __init__(self, *contexts: contextlib.AbstractContextManager[Any]) -> None:
        self._contexts = contexts
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> "_CombinedContext":
        for context in self._contexts:
            self._stack.enter_context(context)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return bool(self._stack.__exit__(exc_type, exc_value, traceback))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise EdgeTAMConfigurationError(
            f"EdgeTAM expects an RGB HxWx3 image, got shape {frame.shape}"
        )
    if frame.size == 0:
        raise EdgeTAMConfigurationError("EdgeTAM RGB image is empty")
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
            raise EdgeTAMConfigurationError(
                "EdgeTAM RGB image must contain finite numeric values"
            )
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _prompt_map(prompts: Sequence[ProjectionPrompt]) -> dict[int, ProjectionPrompt]:
    result: dict[int, ProjectionPrompt] = {}
    for prompt in prompts:
        track_id = int(prompt.track_id)
        if track_id in result:
            raise EdgeTAMConfigurationError(
                f"Duplicate EdgeTAM prompt for track {track_id} in one frame"
            )
        result[track_id] = prompt
    return result


def _copy_prompt(prompt: ProjectionPrompt) -> ProjectionPrompt:
    return ProjectionPrompt(
        track_id=int(prompt.track_id),
        frame_index=int(prompt.frame_index),
        box_xyxy=np.asarray(prompt.box_xyxy, dtype=np.float32).copy(),
        positive_points=np.asarray(prompt.positive_points, dtype=np.float32).copy(),
        negative_points=(
            None
            if prompt.negative_points is None
            else np.asarray(prompt.negative_points, dtype=np.float32).copy()
        ),
        projection_mask=(
            None
            if prompt.projection_mask is None
            else np.asarray(prompt.projection_mask, dtype=bool).copy()
        ),
        re_prompt=bool(prompt.re_prompt),
        reason=str(prompt.reason),
    )


def _select_active_prompts(
    window: Sequence[_WindowFrame],
    active_track_ids: Sequence[int],
) -> dict[int, tuple[int, ProjectionPrompt]]:
    selected: dict[int, tuple[int, ProjectionPrompt]] = {}
    for track_id in active_track_ids:
        candidates = [
            (local_index, frame.prompts[track_id])
            for local_index, frame in enumerate(window)
            if track_id in frame.prompts
        ]
        if not candidates:
            raise EdgeTAMInferenceError(
                f"Confirmed active track {track_id} has no prompt in the rolling window"
            )
        re_prompts = [candidate for candidate in candidates if candidate[1].re_prompt]
        selected[track_id] = re_prompts[-1] if re_prompts else candidates[0]
    return selected


def _finite_points(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return array[np.isfinite(array).all(axis=1)]


def _valid_box(box: np.ndarray) -> bool:
    values = np.asarray(box, dtype=np.float32).reshape(4)
    return bool(
        np.isfinite(values).all()
        and values[2] > values[0]
        and values[3] > values[1]
    )


def _valid_mask(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    array = np.asarray(mask)
    return bool(array.ndim == 2 and array.size > 0)


def _semantic_mask_issue(
    mask: np.ndarray | None,
    prompt: ProjectionPrompt,
    *,
    prompt_frame: int,
    latest_frame: int,
) -> str:
    """Return why propagated output cannot represent the prompted object."""

    if mask is None:
        return "mask_missing"
    candidate = np.asarray(mask, dtype=bool)
    if candidate.ndim != 2:
        return "mask_not_2d"
    if int(np.count_nonzero(candidate)) < 12:
        return "mask_too_small"
    projection = prompt.projection_mask
    if projection is not None:
        support = np.asarray(projection, dtype=bool)
        if candidate.shape != support.shape:
            return "mask_shape_mismatch"
        # A prompt from an older rolling-window frame may have moved before
        # the latest output. Exact-frame prompts, however, must still overlap
        # their measured 3D support or contain a positive seed.
        if int(prompt_frame) == int(latest_frame):
            overlap = int(np.count_nonzero(candidate & support))
            positives = _finite_points(prompt.positive_points)
            positive_hit = False
            if len(positives):
                xy = np.rint(positives).astype(np.int64)
                valid = (
                    (xy[:, 0] >= 0)
                    & (xy[:, 0] < candidate.shape[1])
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < candidate.shape[0])
                )
                xy = xy[valid]
                positive_hit = bool(
                    len(xy)
                    and np.any(candidate[xy[:, 1], xy[:, 0]])
                )
            if overlap == 0 and not positive_hit:
                return "mask_misses_prompt_support"
    return ""


def _prompt_strategies(prompt: ProjectionPrompt) -> tuple[str, ...]:
    strategies: list[str] = []
    if _valid_box(prompt.box_xyxy) and len(_finite_points(prompt.positive_points)) > 0:
        strategies.append("box_points")
    if _valid_mask(prompt.projection_mask):
        strategies.append("mask")
    if _valid_box(prompt.box_xyxy):
        strategies.append("box")
    if not strategies:
        raise EdgeTAMInferenceError(
            f"Track {prompt.track_id} has no valid box, positive point, or mask prompt"
        )
    return tuple(strategies)


def _write_jpeg_window(
    window: Sequence[_WindowFrame],
    directory: Path,
    *,
    jpeg_quality: int,
) -> None:
    if not window:
        raise EdgeTAMInferenceError("Cannot materialize an empty EdgeTAM window")
    expected_shape = window[0].rgb.shape
    for local_index, frame in enumerate(window):
        if frame.rgb.shape != expected_shape:
            raise EdgeTAMInferenceError(
                "All frames in one EdgeTAM video state must have the same shape"
            )
        path = directory / f"{local_index:06d}.jpg"
        bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
        written = cv2.imwrite(
            str(path),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
        if not written or not path.is_file():
            raise EdgeTAMInferenceError(f"Failed to write rolling-window JPEG {path}")


def _to_numpy(value: Any) -> np.ndarray:
    tensor = value
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor)


def _public_video_masks(
    object_ids: Sequence[Any],
    mask_logits: Any,
) -> dict[int, np.ndarray]:
    """Convert the official public video output using its documented >0 threshold."""

    logits = _to_numpy(mask_logits)
    ids = [int(obj_id) for obj_id in object_ids]
    if logits.ndim < 3 or logits.shape[0] != len(ids):
        raise EdgeTAMInferenceError(
            "Unexpected official video output shape: "
            f"object_ids={len(ids)}, mask_logits={logits.shape}"
        )
    masks: dict[int, np.ndarray] = {}
    for index, track_id in enumerate(ids):
        mask = np.asarray(logits[index])
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise EdgeTAMInferenceError(
                f"Unexpected mask shape for track {track_id}: {mask.shape}"
            )
        masks[track_id] = np.asarray(mask > 0.0, dtype=bool)
    return masks


__all__ = [
    "EdgeTAMConfig",
    "EdgeTAMConfigurationError",
    "EdgeTAMError",
    "EdgeTAMInferenceError",
    "EdgeTAMLoadError",
    "EdgeTAMResult",
    "EdgeTAMWrapper",
    "OFFICIAL_CHECKPOINT_SHA256",
    "OFFICIAL_EDGETAM_COMMIT",
]
