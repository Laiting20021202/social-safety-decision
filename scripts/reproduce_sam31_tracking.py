from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from services.dataset_service.video_imports import probe_mp4, sha256_file

SAM3_REPO_COMMIT = "5dd401d1c5c1d5c3eedff06d41b77af824517619"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_trace = output_dir / "error_trace.txt"
    error_trace.write_text("", encoding="utf-8")
    try:
        run(args, output_dir)
    except Exception:
        error_trace.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run(args: argparse.Namespace, output_dir: Path) -> None:
    video_path = Path(args.video).expanduser().resolve()
    info = probe_mp4(video_path)
    assert info["frame_count"] > 30, f"frame_count must be > 30, got {info['frame_count']}"
    assert info["width"] > 0, f"width must be > 0, got {info['width']}"
    assert info["height"] > 0, f"height must be > 0, got {info['height']}"
    assert 0 <= args.frame_index < info["frame_count"], (
        f"prompt frame index {args.frame_index} outside frame_count={info['frame_count']}"
    )

    input_info = {
        **info,
        "video_path": str(video_path),
        "video_hash": sha256_file(video_path),
        "prompt": args.prompt,
        "prompt_frame_index": args.frame_index,
        "max_frames": args.max_frames,
        "sam3_expected_commit": SAM3_REPO_COMMIT,
        "sam3_installed_commit": installed_sam3_commit(args.sam3_repo_dir),
    }
    write_json(output_dir / "input_info.json", input_info)
    session_log = JsonlLogger(output_dir / "session_log.jsonl")

    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    session_log.write(
        "environment",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hf_home": os.getenv("HF_HOME"),
        },
    )

    checkpoint_path = hf_hub_download(
        repo_id=args.repo,
        revision=args.revision,
        filename=args.checkpoint_file,
        token=os.getenv("HF_TOKEN") or None,
        cache_dir=os.getenv("HF_HOME") or None,
    )
    session_log.write(
        "checkpoint",
        {
            "repo": args.repo,
            "revision": args.revision,
            "checkpoint_file": args.checkpoint_file,
            "checkpoint_path": checkpoint_path,
        },
    )

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint_path),
        max_num_objects=args.max_objects,
        multiplex_count=args.multiplex_count,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=True,
    )
    session_log.write(
        "predictor_built",
        {
            "type": type(predictor).__name__,
            "max_num_objects": args.max_objects,
            "multiplex_count": args.multiplex_count,
            "use_fa3": False,
            "compile": False,
            "warm_up": False,
        },
    )

    session_id = start_session(predictor, video_path, session_log)
    prompt_response = add_prompt(
        predictor,
        session_id=session_id,
        frame_index=args.frame_index,
        prompt=args.prompt,
        output_prob_thresh=args.output_prob_thresh,
        session_log=session_log,
    )
    prompt_summary = summarize_frame_response(prompt_response)
    session_log.write("prompt_summary", prompt_summary)
    write_json(output_dir / "prompt_result.json", prompt_summary)
    assert prompt_summary["number_of_objects"] > 0, "prompt produced zero objects"
    assert prompt_summary["masks_shape"][0] > 0, (
        f"prompt masks B=0: {prompt_summary['masks_shape']}"
    )
    assert len(prompt_summary["object_ids"]) > 0, "prompt returned no official object IDs"

    propagated_path = output_dir / "propagated_results.jsonl"
    propagated_path.write_text("", encoding="utf-8")
    propagated_results = []
    overlay_results = []
    with propagated_path.open("a", encoding="utf-8") as file:
        for response in propagate(
            predictor,
            session_id=session_id,
            start_frame_index=args.frame_index,
            max_frames=args.max_frames,
            output_prob_thresh=args.output_prob_thresh,
            session_log=session_log,
        ):
            summary = summarize_frame_response(response)
            propagated_results.append(summary)
            overlay_results.append(overlay_payload(response))
            file.write(json.dumps(summary, sort_keys=True) + "\n")
            file.flush()
    assert len(propagated_results) >= 30, (
        f"propagation returned {len(propagated_results)} frames, expected at least 30"
    )
    expected_ids = set(prompt_summary["object_ids"])
    missing_id_frames = [
        item["frame_index"]
        for item in propagated_results[:30]
        if not expected_ids.intersection(set(item["object_ids"]))
    ]
    assert not missing_id_frames, (
        "official prompt object ID disappeared within first 30 propagated frames: "
        f"{missing_id_frames}"
    )
    write_overlay(video_path, overlay_results[: args.max_frames], output_dir / "overlay.mp4", info)
    session_log.write(
        "complete",
        {
            "propagated_frame_count": len(propagated_results),
            "stable_prompt_object_ids": sorted(expected_ids),
            "overlay_path": str(output_dir / "overlay.mp4"),
        },
    )


def start_session(predictor: Any, video_path: Path, session_log: JsonlLogger) -> str:
    request = {
        "type": "start_session",
        "resource_path": str(video_path),
        "offload_video_to_cpu": True,
    }
    session_log.write("start_session_request", request)
    try:
        response = predictor.handle_request(request)
    except Exception as exc:
        session_log.write(
            "start_session_handle_request_failed",
            {"error": repr(exc), "fallback": "model.init_state"},
        )
        response = start_session_via_init_state(predictor, video_path)
    session_log.write("start_session_response", json_safe(response))
    session_id = response.get("session_id") if isinstance(response, dict) else None
    assert isinstance(session_id, str) and session_id, f"invalid start_session response: {response}"
    return session_id


def start_session_via_init_state(predictor: Any, video_path: Path) -> dict[str, object]:
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    all_states = getattr(predictor, "_all_inference_states", None)
    if init_state is None or not isinstance(all_states, dict):
        raise RuntimeError("predictor does not expose model.init_state or _all_inference_states")
    signature = inspect.signature(init_state)
    candidates: dict[str, object] = {
        "resource_path": str(video_path),
        "video_path": str(video_path),
        "offload_video_to_cpu": True,
        "async_loading_frames": True,
        "input_is_mp4": video_path.suffix.lower() == ".mp4",
    }
    kwargs = {key: value for key, value in candidates.items() if key in signature.parameters}
    inference_state = init_state(**kwargs)
    session_id = str(uuid.uuid4())
    now = time.time()
    all_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": now,
        "last_use_time": now,
    }
    if hasattr(predictor, "_extend_expiration_time"):
        predictor._extend_expiration_time(all_states[session_id])
    return {"session_id": session_id, "start_mode": "model.init_state"}


def add_prompt(
    predictor: Any,
    *,
    session_id: str,
    frame_index: int,
    prompt: str,
    output_prob_thresh: float,
    session_log: JsonlLogger,
) -> dict[str, Any]:
    request = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_index,
        "text": prompt,
        "output_prob_thresh": output_prob_thresh,
    }
    session_log.write("add_prompt_request", request)
    response = predictor.handle_request(request)
    session_log.write("add_prompt_response_summary", summarize_frame_response(response))
    return response


def propagate(
    predictor: Any,
    *,
    session_id: str,
    start_frame_index: int,
    max_frames: int,
    output_prob_thresh: float,
    session_log: JsonlLogger,
) -> Any:
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": start_frame_index,
        "max_frame_num_to_track": max_frames,
        "output_prob_thresh": output_prob_thresh,
    }
    session_log.write("propagate_request", request)
    stream = predictor.handle_stream_request(request)
    for index, response in enumerate(stream):
        summary = summarize_frame_response(response)
        session_log.write("propagate_frame", {"stream_index": index, **summary})
        yield response


def summarize_frame_response(response: dict[str, Any]) -> dict[str, Any]:
    frame_index = int(response.get("frame_index", -1))
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    masks = to_numpy(outputs.get("out_binary_masks"))
    object_ids = to_numpy(outputs.get("out_obj_ids"))
    scores = to_numpy(outputs.get("out_probs"))
    masks_shape = list(masks.shape) if masks is not None else [0]
    object_id_list = (
        [int(value) for value in object_ids.reshape(-1).tolist()]
        if object_ids is not None
        else []
    )
    score_list = (
        [float(value) for value in scores.reshape(-1).tolist()]
        if scores is not None and scores.size <= 64
        else []
    )
    return {
        "frame_index": frame_index,
        "output_keys": sorted(outputs),
        "masks_shape": masks_shape,
        "number_of_objects": int(masks_shape[0]) if masks_shape else 0,
        "object_ids": object_id_list,
        "scores": score_list,
        "batch_dimension": int(masks_shape[0]) if masks_shape else 0,
        "has_non_empty_mask": bool(masks is not None and np.asarray(masks).any()),
        "mask_pixel_counts": mask_pixel_counts(masks),
    }


def mask_pixel_counts(masks: np.ndarray | None) -> list[int]:
    if masks is None:
        return []
    array = np.asarray(masks)
    while array.ndim > 3:
        array = np.squeeze(array, axis=1) if array.shape[1] == 1 else array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        return []
    return [int((mask > 0).sum()) for mask in array]


def write_overlay(
    video_path: Path,
    results: list[dict[str, Any]],
    output_path: Path,
    info: dict[str, int | float],
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {video_path}")
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["native_fps"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open overlay writer: {output_path}")
    frames_by_index = {int(item["frame_index"]): item for item in results}
    max_index = max(frames_by_index) if frames_by_index else -1
    frame_index = 0
    while frame_index <= max_index:
        ok, frame = capture.read()
        if not ok:
            break
        result = frames_by_index.get(frame_index)
        if result is not None:
            frame = draw_overlay_from_summary(frame, result)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()


def draw_overlay_from_summary(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    import cv2

    masks = result.get("masks")
    object_ids = result.get("object_ids")
    if isinstance(masks, np.ndarray):
        array = masks
        if array.ndim == 2:
            array = array[None, ...]
        for index, mask in enumerate(array):
            binary = mask > 0
            if not binary.any():
                continue
            if binary.shape[:2] != frame.shape[:2]:
                binary = cv2.resize(
                    binary.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            color = color_for_index(index)
            color_layer = np.zeros_like(frame)
            color_layer[:, :] = color
            frame[binary] = cv2.addWeighted(frame, 0.55, color_layer, 0.45, 0)[binary]
            contours, _hierarchy = cv2.findContours(
                binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(frame, contours, -1, color, 2)
            if isinstance(object_ids, list) and index < len(object_ids):
                ys, xs = np.where(binary)
                if len(xs) and len(ys):
                    cv2.putText(
                        frame,
                        f"ID {object_ids[index]}",
                        (int(xs.mean()), int(ys.min())),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
    text = f"frame {result['frame_index']} ids={result['object_ids']}"
    cv2.putText(
        frame,
        text,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def overlay_payload(response: dict[str, Any]) -> dict[str, Any]:
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    masks = to_numpy(outputs.get("out_binary_masks"))
    object_ids = to_numpy(outputs.get("out_obj_ids"))
    if masks is not None:
        while masks.ndim > 3:
            masks = np.squeeze(masks, axis=1) if masks.shape[1] == 1 else masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        masks = masks > 0
    return {
        "frame_index": int(response.get("frame_index", -1)),
        "masks": masks,
        "object_ids": [int(value) for value in object_ids.reshape(-1).tolist()]
        if object_ids is not None
        else [],
    }


def color_for_index(index: int) -> tuple[int, int, int]:
    colors = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
    ]
    return colors[index % len(colors)]


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach()
        if str(getattr(tensor, "dtype", "")) in {"torch.bfloat16", "torch.float16"}:
            tensor = tensor.float()
        value = tensor.cpu().numpy()
    return np.asarray(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if hasattr(value, "detach"):
        return json_safe(to_numpy(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def installed_sam3_commit(repo_dir: str) -> str | None:
    path = Path(repo_dir)
    if not path.exists():
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        line = {
            "time": time.time(),
            "event": event,
            "payload": json_safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal official SAM3.1 video tracking repro.")
    parser.add_argument("--video", required=True, help="Absolute or relative path to a real MP4.")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/sam31_reproduction")
    parser.add_argument("--repo", default="facebook/sam3.1")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--checkpoint-file", default="sam3.1_multiplex.pt")
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--sam3-repo-dir", default="/opt/sam3")
    return parser.parse_args()


if __name__ == "__main__":
    main()
