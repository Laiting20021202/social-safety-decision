from __future__ import annotations

import contextlib
import gc
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from packages.common_models import Point2D

SAM3_REPO_COMMIT = "5dd401d1c5c1d5c3eedff06d41b77af824517619"
IMAGE_CHECKPOINT_FILENAME = "sam3.pt"
VIDEO_CHECKPOINT_FILENAME = "sam3.1_multiplex.pt"
SUPPORTED_CLASSES = ("person", "car", "bicycle", "motorcycle", "bus", "truck")
ROAD_PROMPTS = ("road", "sidewalk", "corridor", "walkable path", "traversable ground")


class Sam3RuntimeError(RuntimeError):
    def __init__(self, message: str, *, degraded: bool = False) -> None:
        super().__init__(message)
        self.degraded = degraded


@dataclass
class RuntimeModelState:
    loaded: bool = False
    repo_id: str = ""
    revision: str = ""
    checkpoint_path: str | None = None
    checkpoint_source: str = ""
    device: str = "cuda"
    message: str = "not loaded"

    def as_dict(self) -> dict[str, object]:
        return {
            "loaded": self.loaded,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_source": self.checkpoint_source,
            "device": self.device,
            "message": self.message,
        }


@dataclass
class VideoSessionState:
    session_id: str
    resource_path: str
    scenario_id: str | None = None
    analysis_fps: float = 5.0
    max_objects: int = 6
    created_at: float = field(default_factory=time.time)
    prompt: str | None = None
    prompt_frame_index: int | None = None
    frame_results: dict[int, dict[str, object]] = field(default_factory=dict)


class Sam3ImageRuntime:
    def __init__(self) -> None:
        self.repo_id = os.getenv("SAM3_IMAGE_REPO", "facebook/sam3")
        self.revision = os.getenv("SAM3_IMAGE_REVISION", "main")
        self.checkpoint_path = os.getenv("SAM3_IMAGE_CHECKPOINT") or None
        self.device = _runtime_device()
        self.confidence_threshold = _env_float("SAM3_IMAGE_CONFIDENCE", 0.5)
        self.processor_resolution = _env_int("SAM3_IMAGE_RESOLUTION", 1008)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._state = RuntimeModelState(
            repo_id=self.repo_id,
            revision=self.revision,
            checkpoint_source=self._checkpoint_source(),
            device=self.device,
        )
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def status(self) -> RuntimeModelState:
        self._state.loaded = self.loaded
        return self._state

    def load(self) -> RuntimeModelState:
        with self._lock:
            if self.loaded:
                return self.status()
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model

                checkpoint_path, source, revision = self._resolve_checkpoint()
                model = build_sam3_image_model(
                    device=self.device,
                    checkpoint_path=str(checkpoint_path),
                    load_from_HF=False,
                    compile=_env_bool("SAM3_COMPILE", False),
                )
                self._model = model
                self._processor = Sam3Processor(
                    model,
                    resolution=self.processor_resolution,
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                )
                self._state = RuntimeModelState(
                    loaded=True,
                    repo_id=self.repo_id,
                    revision=revision,
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_source=source,
                    device=self.device,
                    message=f"Loaded SAM 3 image model from {source}.",
                )
                return self._state
            except Exception as exc:
                if _is_cuda_oom(exc):
                    self.unload(clear_cache=True)
                    raise Sam3RuntimeError(
                        f"SAM 3 image model CUDA OOM; model was unloaded ({exc}).",
                        degraded=True,
                    ) from exc
                self._state.message = f"SAM 3 image model unavailable: {exc}"
                raise Sam3RuntimeError(self._state.message) from exc

    def unload(self, *, clear_cache: bool = True) -> RuntimeModelState:
        with self._lock:
            self._processor = None
            self._model = None
            gc.collect()
            if clear_cache:
                empty_cuda_cache()
            self._state.loaded = False
            self._state.message = "image model unloaded"
            return self._state

    def segment_image(
        self,
        image_path: Path,
        prompts: list[str] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            self.load()
            assert self._processor is not None
            prompts = prompts or [*SUPPORTED_CLASSES, *ROAD_PROMPTS]
            prompts = [_normalize_prompt(prompt) for prompt in prompts]
            image = Image.open(image_path).convert("RGB")
            try:
                with _sam3_image_autocast(self.device):
                    inference_state = self._processor.set_image(image)
                    detections: list[dict[str, object]] = []
                    for prompt in prompts:
                        output = self._processor.set_text_prompt(
                            state=inference_state,
                            prompt=prompt,
                        )
                        detections.extend(
                            _normalize_image_outputs(
                                output,
                                prompt=prompt,
                                image_width=image.width,
                                image_height=image.height,
                                model_state=self._state,
                            )
                        )
                return {
                    "source": "sam3",
                    "task": "image_concept_segmentation",
                    "image_width": image.width,
                    "image_height": image.height,
                    "detections": detections,
                    "model": self._state.as_dict(),
                    "vram": vram_usage(),
                }
            except Exception as exc:
                if _is_cuda_oom(exc):
                    self.unload(clear_cache=True)
                    raise Sam3RuntimeError(
                        f"SAM 3 image segmentation CUDA OOM; model was unloaded ({exc}).",
                        degraded=True,
                    ) from exc
                self._state.message = f"SAM 3 image segmentation failed: {exc}"
                raise Sam3RuntimeError(self._state.message, degraded=True) from exc

    def _resolve_checkpoint(self) -> tuple[Path, str, str]:
        if self.checkpoint_path:
            path = Path(self.checkpoint_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"SAM3_IMAGE_CHECKPOINT not found: {path}")
            return path, f"local:{path}", self.revision
        return _download_hf_checkpoint(
            repo_id=self.repo_id,
            revision=self.revision,
            filename=IMAGE_CHECKPOINT_FILENAME,
        )

    def _checkpoint_source(self) -> str:
        if self.checkpoint_path:
            return f"local:{Path(self.checkpoint_path).expanduser()}"
        return f"huggingface:{self.repo_id}@{self.revision}/{IMAGE_CHECKPOINT_FILENAME}"


class Sam31VideoRuntime:
    def __init__(self) -> None:
        self.repo_id = os.getenv("SAM3_VIDEO_REPO", "facebook/sam3.1")
        self.revision = os.getenv("SAM3_VIDEO_REVISION", "main")
        self.checkpoint_path = os.getenv("SAM3_VIDEO_CHECKPOINT") or None
        self.max_num_objects = _env_int("SAM3_MAX_OBJECTS", 6)
        self.multiplex_count = _env_int("SAM3_MULTIPLEX_COUNT", 16)
        self.use_fa3 = _env_bool("SAM3_USE_FA3", False)
        self.use_rope_real = _env_bool("SAM3_USE_ROPE_REAL", False)
        self.compile = _env_bool("SAM3_COMPILE", False)
        self.warm_up = _env_bool("SAM3_WARM_UP", False)
        self.async_loading_frames = _env_bool("SAM3_ASYNC_LOADING_FRAMES", True)
        self.vision_backbone_bf16 = _env_bool("SAM3_VIDEO_VISION_BF16", True)
        self._predictor: Any | None = None
        self._sessions: dict[str, VideoSessionState] = {}
        self._state = RuntimeModelState(
            repo_id=self.repo_id,
            revision=self.revision,
            checkpoint_source=self._checkpoint_source(),
            device="cuda",
        )
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def status(self) -> RuntimeModelState:
        self._state.loaded = self.loaded
        return self._state

    def configure(self, *, max_num_objects: int | None = None) -> None:
        with self._lock:
            if max_num_objects is None or max_num_objects == self.max_num_objects:
                return
            if max_num_objects < 1:
                raise Sam3RuntimeError("max_num_objects must be at least 1.")
            if max_num_objects > 6:
                raise Sam3RuntimeError("max_num_objects must be <= 6 for RTX 4060 safety.")
            if self.loaded:
                self.unload(clear_cache=True)
            self.max_num_objects = max_num_objects

    def load(self) -> RuntimeModelState:
        with self._lock:
            if self.loaded:
                return self.status()
            if not _cuda_available():
                self._state.message = "SAM 3.1 video tracking requires CUDA."
                raise Sam3RuntimeError(self._state.message, degraded=True)
            try:
                self._predictor = self._build_predictor()
                self._state.loaded = True
                self._state.message = (
                    "Loaded SAM 3.1 multiplex video predictor "
                    f"max_num_objects={self.max_num_objects}, "
                    f"multiplex_count={self.multiplex_count}, "
                    f"use_fa3={self.use_fa3}, compile={self.compile}, "
                    f"warm_up={self.warm_up}, "
                    f"vision_backbone_bf16={self.vision_backbone_bf16}."
                )
                return self._state
            except Exception as exc:
                if _is_cuda_oom(exc):
                    self.unload(clear_cache=True)
                    raise Sam3RuntimeError(
                        f"SAM 3.1 video model CUDA OOM; model was unloaded ({exc}).",
                        degraded=True,
                    ) from exc
                self._state.message = f"SAM 3.1 video model unavailable: {exc}"
                raise Sam3RuntimeError(self._state.message) from exc

    def unload(self, *, clear_cache: bool = True) -> RuntimeModelState:
        with self._lock:
            if self._predictor is not None:
                for session_id in list(self._sessions):
                    self._close_session_locked(session_id)
                if hasattr(self._predictor, "shutdown"):
                    self._predictor.shutdown()
            self._predictor = None
            self._sessions.clear()
            gc.collect()
            if clear_cache:
                empty_cuda_cache()
            self._state.loaded = False
            self._state.message = "video model unloaded"
            return self._state

    def start_session(
        self,
        resource_path: str,
        session_id: str | None = None,
        *,
        scenario_id: str | None = None,
        analysis_fps: float = 5.0,
    ) -> dict[str, object]:
        with self._lock:
            self.load()
            path = Path(resource_path)
            if not path.is_absolute():
                raise Sam3RuntimeError("start_session requires an absolute MP4 path.")
            if path.suffix.lower() != ".mp4":
                raise Sam3RuntimeError("start_session only accepts an MP4 resource path.")
            if not path.exists():
                raise Sam3RuntimeError(f"MP4 resource path does not exist: {path}")
            assert self._predictor is not None
            try:
                response = self._start_predictor_session(path=path, session_id=session_id)
            except Exception as exc:
                self._handle_operation_error(exc)
            official_session_id = str(response["session_id"])
            self._sessions[official_session_id] = VideoSessionState(
                session_id=official_session_id,
                resource_path=str(path),
                scenario_id=scenario_id,
                analysis_fps=analysis_fps,
                max_objects=self.max_num_objects,
            )
            return {
                "source": "sam3",
                "task": "sam3.1_video_tracking",
                "session_id": official_session_id,
                "resource_path": str(path),
                "scenario_id": scenario_id,
                "analysis_fps": analysis_fps,
                "max_objects": self.max_num_objects,
                "model": self._state.as_dict(),
                "vram": vram_usage(),
            }

    def add_prompt(
        self,
        session_id: str,
        prompt: str,
        frame_index: int = 0,
        output_prob_thresh: float = 0.5,
    ) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            prompt = _normalize_prompt(prompt)
            if prompt not in SUPPORTED_CLASSES:
                raise Sam3RuntimeError(
                    f"Unsupported video text prompt {prompt!r}; supported prompts: "
                    f"{', '.join(SUPPORTED_CLASSES)}."
                )
            assert self._predictor is not None
            try:
                response = self._predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": prompt,
                        "output_prob_thresh": output_prob_thresh,
                    }
                )
            except Exception as exc:
                self._handle_operation_error(exc)
            result = _normalize_video_frame_response(
                response,
                prompt=prompt,
                session=session,
                model_state=self._state,
            )
            session.prompt = prompt
            session.prompt_frame_index = frame_index
            result_frame_index = _coerce_int(result.get("frame_index"), frame_index)
            session.frame_results[result_frame_index] = result
            return result

    def propagate(
        self,
        session_id: str,
        propagation_direction: Literal["both", "forward", "backward"] = "forward",
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float = 0.5,
        include_frames: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            assert self._predictor is not None
            frames: list[dict[str, object]] = []
            try:
                stream = self._predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": propagation_direction,
                        "start_frame_index": start_frame_index,
                        "max_frame_num_to_track": max_frame_num_to_track,
                        "output_prob_thresh": output_prob_thresh,
                    }
                )
                for response in stream:
                    result = _normalize_video_frame_response(
                        response,
                        prompt=session.prompt or "unknown",
                        session=session,
                        model_state=self._state,
                    )
                    result_frame_index = _coerce_int(result.get("frame_index"), 0)
                    session.frame_results[result_frame_index] = result
                    if include_frames:
                        frames.append(result)
            except Exception as exc:
                self._handle_operation_error(exc)
            return {
                "source": "sam3",
                "task": "sam3.1_video_tracking",
                "session_id": session_id,
                "status": "ok",
                "processed_frame_count": len(session.frame_results),
                "frame_indices": sorted(session.frame_results),
                "frames": frames,
                "model": self._state.as_dict(),
                "vram": vram_usage(),
            }

    def get_frame_result(self, session_id: str, frame_index: int) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            result = session.frame_results.get(frame_index)
            if result is not None:
                return result
            return {
                "source": "sam3",
                "task": "sam3.1_video_tracking",
                "status": "pending",
                "session_id": session_id,
                "frame_index": frame_index,
                "detections": [],
                "message": "Frame has not been propagated in this session.",
                "model": self._state.as_dict(),
            }

    def close_session(self, session_id: str) -> dict[str, object]:
        with self._lock:
            return self._close_session_locked(session_id)

    def session_status(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self._require_session(session_id)
            return {
                "source": "sam3",
                "task": "sam3.1_video_tracking",
                "status": "ok",
                "session_id": session.session_id,
                "resource_path": session.resource_path,
                "scenario_id": session.scenario_id,
                "analysis_fps": session.analysis_fps,
                "max_objects": session.max_objects,
                "prompt": session.prompt,
                "prompt_frame_index": session.prompt_frame_index,
                "cached_frame_count": len(session.frame_results),
                "frame_indices": sorted(session.frame_results),
                "created_at": session.created_at,
                "model": self._state.as_dict(),
                "vram": vram_usage(),
            }

    def sessions(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "session_id": session.session_id,
                    "resource_path": session.resource_path,
                    "scenario_id": session.scenario_id,
                    "analysis_fps": session.analysis_fps,
                    "max_objects": session.max_objects,
                    "prompt": session.prompt,
                    "prompt_frame_index": session.prompt_frame_index,
                    "cached_frame_count": len(session.frame_results),
                    "created_at": session.created_at,
                }
                for session in self._sessions.values()
            ]

    def _start_predictor_session(self, *, path: Path, session_id: str | None) -> dict[str, object]:
        """Start a SAM 3.1 session while filtering args for this pinned package."""
        assert self._predictor is not None
        import inspect
        import uuid

        model = getattr(self._predictor, "model", None)
        init_state = getattr(model, "init_state", None)
        all_states = getattr(self._predictor, "_all_inference_states", None)
        if init_state is None or not isinstance(all_states, dict):
            return self._predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(path),
                    "session_id": session_id,
                    "offload_video_to_cpu": True,
                }
            )

        signature = inspect.signature(init_state)
        candidates: dict[str, object] = {
            "offload_video_to_cpu": True,
            "async_loading_frames": self.async_loading_frames,
            "input_is_mp4": path.suffix.lower() == ".mp4",
        }
        if "resource_path" in signature.parameters:
            candidates["resource_path"] = str(path)
        elif "video_path" in signature.parameters:
            candidates["video_path"] = str(path)
        else:
            return self._predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(path),
                    "session_id": session_id,
                    "offload_video_to_cpu": True,
                }
            )

        init_kwargs = {
            key: value for key, value in candidates.items() if key in signature.parameters
        }
        inference_state = init_state(**init_kwargs)
        official_session_id = session_id or str(uuid.uuid4())
        now = time.time()
        all_states[official_session_id] = {
            "state": inference_state,
            "session_id": official_session_id,
            "start_time": now,
            "last_use_time": now,
        }
        if hasattr(self._predictor, "_extend_expiration_time"):
            self._predictor._extend_expiration_time(all_states[official_session_id])
        return {"session_id": official_session_id}

    def _add_text_prompt_direct(
        self,
        *,
        session_id: str,
        frame_index: int,
        prompt: str,
        output_prob_thresh: float,
    ) -> dict[str, object]:
        """Call the official SAM 3.1 model while avoiding the base wrapper's BF16 text bug."""
        assert self._predictor is not None
        import inspect

        import torch

        all_states = getattr(self._predictor, "_all_inference_states", None)
        model = getattr(self._predictor, "model", None)
        model_add_prompt = getattr(model, "add_prompt", None)
        if not isinstance(all_states, dict) or model_add_prompt is None:
            return self._predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "text": prompt,
                    "output_prob_thresh": output_prob_thresh,
                }
            )
        predictor_session = all_states.get(session_id)
        if not isinstance(predictor_session, dict):
            raise Sam3RuntimeError(f"Cannot find SAM 3.1 video session: {session_id}")
        if hasattr(self._predictor, "_extend_expiration_time"):
            self._predictor._extend_expiration_time(predictor_session)
        inference_state = predictor_session["state"]
        kwargs: dict[str, object] = {
            "inference_state": inference_state,
            "frame_idx": frame_index,
            "text_str": prompt,
            "clear_old_points": True,
            "points": None,
            "point_labels": None,
            "boxes_xywh": None,
            "box_labels": None,
            "clear_old_boxes": True,
            "output_prob_thresh": output_prob_thresh,
        }
        signature = inspect.signature(model_add_prompt)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=False):
                out_frame_index, outputs = model_add_prompt(**filtered_kwargs)
        return {"frame_index": out_frame_index, "outputs": outputs}

    def _build_predictor(self) -> Any:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        checkpoint_path, source, revision = self._resolve_checkpoint()
        self._state.checkpoint_path = str(checkpoint_path)
        self._state.checkpoint_source = source
        self._state.revision = revision
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            max_num_objects=self.max_num_objects,
            multiplex_count=self.multiplex_count,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
            compile=self.compile,
            warm_up=self.warm_up,
            async_loading_frames=self.async_loading_frames,
        )
        if not self.use_fa3:
            self._disable_predictor_bf16_context(predictor)
        if self.vision_backbone_bf16:
            self._cast_video_vision_backbone_to_bf16(predictor)
        return predictor

    def _disable_predictor_bf16_context(self, predictor: Any) -> None:
        context = getattr(predictor, "bf16_context", None)
        if context is None:
            return
        try:
            context.__exit__(None, None, None)
        except RuntimeError:
            return
        predictor.bf16_context = contextlib.nullcontext()

    def _cast_video_vision_backbone_to_bf16(self, predictor: Any) -> None:
        import torch

        model = getattr(predictor, "model", None)
        detector = getattr(model, "detector", None)
        backbone = getattr(detector, "backbone", None)
        vision_backbone = getattr(backbone, "vision_backbone", None)
        if vision_backbone is None:
            return
        _cast_module_floating_dtype(vision_backbone, torch.bfloat16)

    def _resolve_checkpoint(self) -> tuple[Path, str, str]:
        if self.checkpoint_path:
            path = Path(self.checkpoint_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"SAM3_VIDEO_CHECKPOINT not found: {path}")
            return path, f"local:{path}", self.revision
        return _download_hf_checkpoint(
            repo_id=self.repo_id,
            revision=self.revision,
            filename=VIDEO_CHECKPOINT_FILENAME,
        )

    def _checkpoint_source(self) -> str:
        if self.checkpoint_path:
            return f"local:{Path(self.checkpoint_path).expanduser()}"
        return f"huggingface:{self.repo_id}@{self.revision}/{VIDEO_CHECKPOINT_FILENAME}"

    def _require_session(self, session_id: str) -> VideoSessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise Sam3RuntimeError(f"Cannot find SAM 3.1 video session: {session_id}")
        return session

    def _close_session_locked(self, session_id: str) -> dict[str, object]:
        response: dict[str, object] = {"is_success": True}
        if self._predictor is not None:
            try:
                raw = self._predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                        "clear_cache_threshold": 0,
                    }
                )
                if isinstance(raw, dict):
                    response.update(_json_safe(raw))
            except Exception as exc:
                if _is_cuda_oom(exc):
                    self.unload(clear_cache=True)
                    raise Sam3RuntimeError(
                        f"SAM 3.1 close_session CUDA OOM; model was unloaded ({exc}).",
                        degraded=True,
                    ) from exc
                response["is_success"] = False
                response["message"] = str(exc)
        self._sessions.pop(session_id, None)
        response.update(
            {
                "source": "sam3",
                "task": "sam3.1_video_tracking",
                "session_id": session_id,
                "vram": vram_usage(),
            }
        )
        return response

    def _handle_operation_error(self, exc: Exception) -> None:
        if _is_cuda_oom(exc):
            self.unload(clear_cache=True)
            raise Sam3RuntimeError(
                f"SAM 3.1 video tracking CUDA OOM; model/session unloaded ({exc}).",
                degraded=True,
            ) from exc
        raise Sam3RuntimeError(str(exc)) from exc


class Sam3RuntimeManager:
    def __init__(self) -> None:
        self.image = Sam3ImageRuntime()
        self.video = Sam31VideoRuntime()
        self._lock = threading.RLock()

    def load_image_model(self) -> RuntimeModelState:
        with self._lock:
            if self.video.loaded:
                self.video.unload(clear_cache=True)
            return self.image.load()

    def unload_image_model(self) -> RuntimeModelState:
        return self.image.unload(clear_cache=True)

    def load_video_model(self) -> RuntimeModelState:
        with self._lock:
            if self.image.loaded:
                self.image.unload(clear_cache=True)
            return self.video.load()

    def unload_video_model(self) -> RuntimeModelState:
        return self.video.unload(clear_cache=True)

    def segment_image(
        self,
        image_path: Path,
        prompts: list[str] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self.video.loaded:
                self.video.unload(clear_cache=True)
            return self.image.segment_image(image_path, prompts)

    def start_video_session(
        self,
        resource_path: str,
        session_id: str | None = None,
        *,
        scenario_id: str | None = None,
        analysis_fps: float = 5.0,
        max_objects: int | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self.image.loaded:
                self.image.unload(clear_cache=True)
            self.video.configure(max_num_objects=max_objects)
            return self.video.start_session(
                resource_path=resource_path,
                session_id=session_id,
                scenario_id=scenario_id,
                analysis_fps=analysis_fps,
            )

    def status(self) -> dict[str, object]:
        return {
            "sam3_package_commit": SAM3_REPO_COMMIT,
            "cuda": cuda_info(),
            "vram": vram_usage(),
            "image_model": self.image.status().as_dict(),
            "video_model": self.video.status().as_dict(),
            "video_sessions": self.video.sessions(),
            "safety": {
                "models_mutually_exclusive": True,
                "max_num_objects": self.video.max_num_objects,
                "use_fa3": self.video.use_fa3,
                "compile": self.video.compile,
                "warm_up": self.video.warm_up,
            },
        }


def _normalize_image_outputs(
    output: dict[str, Any],
    *,
    prompt: str,
    image_width: int,
    image_height: int,
    model_state: RuntimeModelState,
) -> list[dict[str, object]]:
    masks = _to_numpy(output.get("masks"))
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))
    if masks is None:
        return []
    if masks.ndim == 2:
        masks = masks[None, ...]
    detections: list[dict[str, object]] = []
    for index in range(masks.shape[0]):
        mask = _as_binary_mask(masks[index])
        if not mask.any():
            continue
        bbox = (
            _box_xyxy_to_tuple(boxes[index])
            if boxes is not None and index < len(boxes)
            else _bbox_from_mask(mask)
        )
        score = _score_at(scores, index)
        detections.append(
            _detection_from_mask(
                mask,
                bbox=bbox,
                class_name=_class_from_prompt(prompt),
                confidence=score,
                label=prompt,
                object_id=None,
                frame_index=None,
                model_state=model_state,
                segmentation_role="image_concept_segmentation",
                image_width=image_width,
                image_height=image_height,
            )
        )
    return detections


def _normalize_video_frame_response(
    response: dict[str, Any],
    *,
    prompt: str,
    session: VideoSessionState,
    model_state: RuntimeModelState,
) -> dict[str, object]:
    frame_index = int(response["frame_index"])
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    masks = _to_numpy(outputs.get("out_binary_masks"))
    object_ids = _to_numpy(outputs.get("out_obj_ids"))
    boxes = _to_numpy(outputs.get("out_boxes_xywh"))
    scores = _to_numpy(outputs.get("out_probs"))
    if masks is None:
        masks = np.zeros((0, 0, 0), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    detections: list[dict[str, object]] = []
    for index in range(masks.shape[0]):
        if object_ids is None or index >= len(object_ids):
            continue
        object_id = int(object_ids[index])
        mask = _as_binary_mask(masks[index])
        if not mask.any():
            continue
        height, width = mask.shape
        bbox = (
            _box_xywh_relative_to_xyxy(boxes[index], width=width, height=height)
            if boxes is not None and index < len(boxes)
            else _bbox_from_mask(mask)
        )
        detections.append(
            _detection_from_mask(
                mask,
                bbox=bbox,
                class_name=_class_from_prompt(prompt),
                confidence=_score_at(scores, index),
                label=prompt,
                object_id=object_id,
                frame_index=frame_index,
                model_state=model_state,
                segmentation_role="sam3.1_video_tracking",
                image_width=width,
                image_height=height,
            )
        )
    return {
        "source": "sam3",
        "task": "sam3.1_video_tracking",
        "status": "ok",
        "session_id": session.session_id,
        "resource_path": session.resource_path,
        "frame_index": frame_index,
        "prompt": prompt,
        "image_width": int(masks.shape[-1]) if masks.ndim == 3 else 0,
        "image_height": int(masks.shape[-2]) if masks.ndim == 3 else 0,
        "detections": detections,
        "model": model_state.as_dict(),
    }


def _detection_from_mask(
    mask: NDArray[np.bool_],
    *,
    bbox: tuple[float, float, float, float],
    class_name: str,
    confidence: float,
    label: str,
    object_id: int | None,
    frame_index: int | None,
    model_state: RuntimeModelState,
    segmentation_role: str,
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    polygon = _mask_to_contour_polygon(mask)
    centroid = _mask_centroid(mask)
    ground = _mask_ground_contact(mask)
    area = int(mask.sum())
    payload: dict[str, object] = {
        "object_id": object_id,
        "track_id": object_id,
        "class_name": class_name,
        "confidence": confidence,
        "bounding_box": bbox,
        "mask_rle": _mask_to_coco_rle(mask),
        "mask_polygon": [point.model_dump(mode="json") for point in polygon],
        "mask_area": area,
        "mask_format": "coco_rle_uncompressed",
        "segmentation_type": "binary_mask",
        "segmentation_role": segmentation_role,
        "mask_polygon_kind": "convex_hull_of_mask_boundary",
        "ground_contact_point": ground.model_dump(mode="json") if ground else None,
        "centroid": centroid.model_dump(mode="json") if centroid else None,
        "source": "sam3",
        "label": label,
        "frame_index": frame_index,
        "image_width": image_width,
        "image_height": image_height,
        "model_revision": model_state.revision,
        "checkpoint_source": model_state.checkpoint_source,
    }
    if object_id is None:
        payload["track_id"] = None
    return payload


def _download_hf_checkpoint(repo_id: str, revision: str, filename: str) -> tuple[Path, str, str]:
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN") or None
    cache_dir = os.getenv("HF_HOME") or None
    hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename="config.json",
        token=token,
        cache_dir=cache_dir,
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            token=token,
            cache_dir=cache_dir,
        )
    )
    snapshot_revision = _snapshot_revision_from_path(checkpoint) or revision
    return checkpoint, f"huggingface:{repo_id}@{snapshot_revision}/{filename}", snapshot_revision


def _snapshot_revision_from_path(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _to_numpy(value: Any) -> NDArray[Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach()
        if str(getattr(tensor, "dtype", "")) in {"torch.bfloat16", "torch.float16"}:
            tensor = tensor.float()
        value = tensor.cpu().numpy()
    return np.asarray(value)


def _cast_module_floating_dtype(module: Any, dtype: Any) -> None:
    for child in module.modules():
        for parameter in child._parameters.values():
            if parameter is not None and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=dtype)
        for name, buffer in child._buffers.items():
            if buffer is not None and buffer.is_floating_point():
                child._buffers[name] = buffer.to(dtype=dtype)


def _as_binary_mask(mask: NDArray[Any]) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    while array.ndim > 2:
        array = np.squeeze(array, axis=0) if array.shape[0] == 1 else array[0]
    return array > 0


def _mask_to_coco_rle(mask: NDArray[np.bool_]) -> dict[str, object]:
    pixels = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = value
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _mask_to_contour_polygon(mask: NDArray[np.bool_], max_points: int = 80) -> list[Point2D]:
    boundary = _boundary_points(mask)
    if len(boundary) < 3:
        return []
    hull = _convex_hull(boundary)
    if len(hull) > max_points:
        stride = max(1, math.ceil(len(hull) / max_points))
        hull = hull[::stride]
    return [Point2D(x=float(x), y=float(y)) for x, y in hull]


def _boundary_points(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = center & ~interior
    ys, xs = np.where(boundary)
    if len(xs) > 4000:
        stride = math.ceil(len(xs) / 4000)
        xs = xs[::stride]
        ys = ys[::stride]
    return sorted(set(zip(xs.tolist(), ys.tolist(), strict=True)))


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    def cross(origin: tuple[int, int], left: tuple[int, int], right: tuple[int, int]) -> int:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    sorted_points = sorted(set(points))
    if len(sorted_points) <= 1:
        return sorted_points
    lower: list[tuple[int, int]] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _bbox_from_mask(mask: NDArray[np.bool_]) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _box_xyxy_to_tuple(box: NDArray[Any]) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size < 4:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _box_xywh_relative_to_xyxy(
    box: NDArray[Any],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size < 4:
        return (0.0, 0.0, 0.0, 0.0)
    x, y, w, h = values[:4]
    return (
        float(x * width),
        float(y * height),
        float((x + w) * width),
        float((y + h) * height),
    )


def _mask_centroid(mask: NDArray[np.bool_]) -> Point2D | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return Point2D(x=float(xs.mean()), y=float(ys.mean()))


def _mask_ground_contact(mask: NDArray[np.bool_]) -> Point2D | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    bottom_y = int(ys.max())
    bottom_xs = xs[ys >= max(0, bottom_y - 2)]
    x = float(bottom_xs.mean()) if len(bottom_xs) else float(xs.mean())
    return Point2D(x=x, y=float(bottom_y))


def _score_at(scores: NDArray[Any] | None, index: int) -> float:
    if scores is None or index >= len(scores):
        return 0.5
    return max(0.0, min(1.0, float(np.asarray(scores[index]).reshape(-1)[0])))


def _class_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    for class_name in SUPPORTED_CLASSES:
        if class_name in lowered:
            return class_name
    for class_name in ROAD_PROMPTS:
        if class_name in lowered:
            return class_name
    return "unknown"


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _sam3_image_autocast(device: str) -> contextlib.AbstractContextManager[object]:
    if not device.startswith("cuda") or not _env_bool("SAM3_IMAGE_AUTOCAST", True):
        return contextlib.nullcontext()
    import torch

    dtype_name = os.getenv("SAM3_IMAGE_AMP_DTYPE", "bfloat16").strip().lower()
    dtype = torch.float16 if dtype_name in {"fp16", "float16"} else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _runtime_device() -> str:
    return "cuda" if _cuda_available() else "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _is_cuda_oom(exc: Exception) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return "cuda out of memory" in str(exc).lower()


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def vram_usage() -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "used_bytes": int(total_bytes - free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "device_name": torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        return {"cuda_available": False, "message": str(exc)}


def cuda_info() -> dict[str, object]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {"available": False, "message": str(exc)}


def write_upload_to_temp(data: bytes, suffix: str = ".png") -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return Path(handle.name)
