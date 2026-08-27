"""Binary-protocol worker around the official MASt3R-SLAM code.

This file intentionally avoids imports from ``realtime_safety`` so it can run
inside MASt3R-SLAM's dedicated virtual environment.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HEADER = struct.Struct("<Q")


class LocalKeyframes:
    """Small in-process equivalent of upstream's preallocated shared store."""

    def __init__(self) -> None:
        self.frames = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[int(index)]

    def __setitem__(self, index, value):
        self.frames[int(index)] = value

    def append(self, frame):
        self.frames.append(frame)

    def pop_last(self):
        return self.frames.pop()

    def last_keyframe(self):
        return self.frames[-1] if self.frames else None

    def update_T_WCs(self, poses, indices) -> None:
        import lietorch

        for offset, index in enumerate(indices.detach().cpu().tolist()):
            self.frames[int(index)].T_WC = lietorch.Sim3(
                poses.data[offset].clone()
            )


class SlamEngine:
    def __init__(self, args) -> None:
        root = Path(args.root).resolve()
        sys.path.insert(0, str(root))

        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_mast3r, load_retriever
        from mast3r_slam.tracker import FrameTracker

        load_config(str(Path(args.config).resolve()))
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        # PyTorch 2.6+ defaults torch.load() to weights_only=True. The official
        # MASt3R checkpoint stores its model arguments in this one standard-
        # library type, so allowlist it without disabling safe loading.
        torch.serialization.add_safe_globals([argparse.Namespace])
        self.torch = torch
        self.config = config
        self.image_size = int(args.image_size)
        self.confidence_threshold = float(args.confidence_threshold)
        self.max_points = int(args.max_points)
        self.voxel_size = float(args.voxel_size)
        self.device = args.device if torch.cuda.is_available() else "cpu"
        if not self.device.startswith("cuda"):
            raise RuntimeError("Official MASt3R-SLAM requires an NVIDIA CUDA GPU")
        self.model = load_mast3r(str(Path(args.checkpoint).resolve()), self.device).eval()
        self.loop_closure = not args.no_loop_closure
        self.retrieval_checkpoint = (
            str(Path(args.retrieval_checkpoint).resolve())
            if self.loop_closure
            else None
        )
        self.retriever = (
            load_retriever(self.model, self.retrieval_checkpoint, self.device)
            if self.loop_closure
            else None
        )
        self.FactorGraph = FactorGraph
        self.FrameTracker = FrameTracker
        self.reset()

    def reset(self) -> None:
        self.keyframes = LocalKeyframes()
        self.tracker = self.FrameTracker(self.model, self.keyframes, self.device)
        self.factor_graph = self.FactorGraph(
            self.model, self.keyframes, K=None, device=self.device
        )
        if self.retriever is not None:
            # Keep the loaded retrieval network/codebook and clear only its
            # per-sequence inverted index. Reloading weights on every source
            # restart is slow and can briefly double GPU allocations.
            self.retriever.ivf_builder = self.retriever.asmk.create_ivf_builder()
            self.retriever.kf_counter = 0
            self.retriever.kf_ids = []
        self.current_frame = None
        self.sequence_index = 0

    def infer(self, message: dict) -> dict:
        import lietorch
        from mast3r_slam.frame import create_frame
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        started = time.perf_counter()
        rgb = np.asarray(message["rgb"], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        pose = (
            lietorch.Sim3.Identity(1, device=self.device)
            if self.current_frame is None
            else self.current_frame.T_WC
        )
        frame = create_frame(
            int(message["frame_index"]),
            rgb.astype(np.float32) / 255.0,
            pose,
            img_size=self.image_size,
            device=self.device,
        )
        added_as_relocalization = False
        if not self.keyframes.frames:
            points, confidence = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(points, confidence)
            self.keyframes.append(frame)
            if self.retriever is not None:
                self.retriever.update(
                    frame,
                    add_after_query=True,
                    k=self.config["retrieval"]["k"],
                    min_thresh=self.config["retrieval"]["min_thresh"],
                )
        else:
            add_new_keyframe, _match_info, try_relocalize = self.tracker.track(frame)
            if try_relocalize:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                added_as_relocalization = self._relocalize(frame)
                if not added_as_relocalization:
                    # Keep the last globally registered map and report the
                    # tracking loss without inventing a world pose.
                    raise RuntimeError(
                        f"MASt3R-SLAM tracking lost at frame {message['frame_index']}"
                    )
                add_new_keyframe = False
            if add_new_keyframe:
                self.keyframes.append(frame)
                self._optimize_new_keyframe(len(self.keyframes) - 1)
        self.current_frame = frame
        self.sequence_index += 1
        return self._result(frame, message, started)

    def _optimize_new_keyframe(self, index: int) -> None:
        candidates = [index - 1] if index > 0 else []
        if self.retriever is not None:
            retrieved = self.retriever.update(
                self.keyframes[index],
                add_after_query=True,
                k=self.config["retrieval"]["k"],
                min_thresh=self.config["retrieval"]["min_thresh"],
            )
            candidates.extend(retrieved)
        candidates = sorted({int(value) for value in candidates if int(value) != index})
        if not candidates:
            return
        self.factor_graph.add_factors(
            candidates,
            [index] * len(candidates),
            self.config["local_opt"]["min_match_frac"],
        )
        self.factor_graph.solve_GN_rays()

    def _relocalize(self, frame) -> bool:
        if self.retriever is None:
            return False
        candidates = self.retriever.update(
            frame,
            add_after_query=False,
            k=self.config["retrieval"]["k"],
            min_thresh=self.config["retrieval"]["min_thresh"],
        )
        if not candidates:
            return False
        self.keyframes.append(frame)
        index = len(self.keyframes) - 1
        success = self.factor_graph.add_factors(
            [index] * len(candidates),
            list(candidates),
            self.config["reloc"]["min_match_frac"],
            is_reloc=self.config["reloc"]["strict"],
        )
        if not success:
            self.keyframes.pop_last()
            return False
        frame.T_WC = self.keyframes[int(candidates[0])].T_WC.clone()
        self.keyframes[index] = frame
        self.factor_graph.solve_GN_rays()
        self.retriever.update(
            frame,
            add_after_query=True,
            k=self.config["retrieval"]["k"],
            min_thresh=self.config["retrieval"]["min_thresh"],
        )
        return True

    def _result(self, frame, message: dict, started: float) -> dict:
        pointmap = (
            frame.T_WC.act(frame.X_canon)
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(int(frame.img_shape[0, 0]), int(frame.img_shape[0, 1]), 3)
        )
        dense_confidence = (
            frame.get_average_conf()
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(pointmap.shape[:2])
        )
        map_points, map_colors, map_confidence = [], [], []
        for keyframe in self.keyframes.frames:
            points = (
                keyframe.T_WC.act(keyframe.X_canon)
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )
            confidence = (
                keyframe.get_average_conf()
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            colors = (
                keyframe.uimg.detach().cpu().numpy().reshape(-1, 3) * 255.0
            ).clip(0, 255).astype(np.uint8)
            valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
            valid &= confidence >= self.confidence_threshold
            map_points.append(points[valid])
            map_colors.append(colors[valid])
            map_confidence.append(confidence[valid])
        points = np.concatenate(map_points, axis=0)
        colors = np.concatenate(map_colors, axis=0)
        confidence = np.concatenate(map_confidence, axis=0)
        points, colors, confidence = _voxel_downsample(
            points,
            colors,
            confidence,
            float(message.get("voxel_size", self.voxel_size)),
            int(message.get("max_points", self.max_points)),
        )
        return {
            "ok": True,
            "frame_index": int(message["frame_index"]),
            "timestamp": float(message["timestamp"]),
            "anchor_frame_index": int(self.keyframes.frames[-1].frame_id),
            "pointmap": pointmap,
            "dense_confidence": dense_confidence,
            "points": points,
            "colors": colors,
            "confidence": confidence,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }


def _voxel_downsample(points, colors, confidence, voxel_size, max_points):
    if len(points) == 0:
        return points, colors, confidence
    if voxel_size > 0:
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        indices.sort()
        points, colors, confidence = points[indices], colors[indices], confidence[indices]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors, confidence = points[indices], colors[indices], confidence[indices]
    return (
        points.astype(np.float32, copy=False),
        colors.astype(np.uint8, copy=False),
        confidence.astype(np.float32, copy=False),
    )


def _read_exact(stream, size):
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(stream):
    (size,) = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    return pickle.loads(_read_exact(stream, size))


def _write_message(stream, message):
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--retrieval-checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--no-loop-closure", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol_out = sys.stdout.buffer
    # Upstream uses print() for tracking diagnostics. Keep those messages away
    # from the binary stdout protocol while still showing them to the operator.
    sys.stdout = sys.stderr
    try:
        engine = SlamEngine(args)
        _write_message(protocol_out, {"ok": True, "event": "ready"})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _write_message(protocol_out, {"ok": False, "event": "ready", "error": str(exc)})
        return 1
    while True:
        try:
            message = _read_message(sys.stdin.buffer)
        except EOFError:
            return 0
        command = message.get("command")
        if command == "close":
            return 0
        try:
            if command == "reset":
                # A newly started worker is already empty. Avoid immediately
                # reloading the large retrieval model when a source is opened.
                if engine.sequence_index > 0:
                    engine.reset()
                response = {"ok": True}
            elif command == "infer":
                response = engine.infer(message)
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(exc)}
        _write_message(protocol_out, response)


if __name__ == "__main__":
    raise SystemExit(main())
