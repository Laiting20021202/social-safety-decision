from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

from realtime_safety.config import load_config
from realtime_safety.export.pointcloud_export import export_ply
from realtime_safety.export.session_logger import SessionLogger
from realtime_safety.export.video_recorder import VideoRecorder
from realtime_safety.gui.dashboard import Dashboard
from realtime_safety.gui.reconstruction_scene import ReconstructionScene3D
from realtime_safety.gui.scene_3d import Scene3D
from realtime_safety.pipeline.video_source import CameraDetectionError
from realtime_safety.scheduler import RealtimePipeline
from realtime_safety.utils.validation import validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive temporal-depth 4D reconstruction viewer and safety dashboard")
    parser.add_argument("--source", help="Video path, webcam index, or RTSP URL")
    parser.add_argument(
        "--auto-webcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically detect and start the first readable webcam when --source is omitted",
    )
    parser.add_argument("--profile", default="st4rtrack_viewer", help="Profile name or YAML path")
    parser.add_argument("--mode", choices=("reconstruction", "safety"), help="Override the profile application mode")
    parser.add_argument(
        "--people",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable YOLO person masks, 3D person boxes, centers, and direction arrows",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--depth-mode", choices=("video_depth", "st4rtrack", "hybrid", "fast_depth", "rgbd"))
    parser.add_argument("--scale-mode", choices=("relative", "calibrated", "rgbd"))
    parser.add_argument("--manual-scale", type=float)
    parser.add_argument("--focal-x", type=float, help="Webcam intrinsic fx in pixels for 3D reprojection")
    parser.add_argument("--focal-y", type=float, help="Webcam intrinsic fy in pixels for 3D reprojection")
    parser.add_argument("--principal-x", type=float, help="Webcam intrinsic cx in pixels")
    parser.add_argument("--principal-y", type=float, help="Webcam intrinsic cy in pixels")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Do not launch Viser")
    parser.add_argument("--max-frames", type=int, help="Stop capture after this many frames")
    parser.add_argument("--duration", type=float, help="Stop after wall-clock seconds")
    parser.add_argument("--exit-on-end", action="store_true", help="Exit GUI when finite source ends")
    parser.add_argument("--output-dir", help="Session output directory (default: sessions/<timestamp>)")
    parser.add_argument("--no-log", action="store_true", help="Disable streaming JSONL/trajectory CSV")
    parser.add_argument("--record", action="store_true", help="Record annotated MP4")
    parser.add_argument("--export-ply", action="store_true", help="Export the final visible point cloud on exit")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    config = load_config(args.profile)
    if args.mode:
        config.mode = args.mode
    if args.people is not None:
        config.people_overlay = args.people
    if args.device:
        config.device = args.device
    if args.depth_mode:
        config.reconstruction.depth_mode = args.depth_mode
    if args.scale_mode:
        config.scale_mode = args.scale_mode
    if args.manual_scale is not None:
        if args.manual_scale <= 0:
            raise ValueError("--manual-scale must be positive")
        config.manual_scale = args.manual_scale
        config.scale_mode = "calibrated"
    if args.focal_x is not None:
        config.reconstruction.focal_length_x = args.focal_x
    if args.focal_y is not None:
        config.reconstruction.focal_length_y = args.focal_y
    if args.principal_x is not None:
        config.reconstruction.principal_point_x = args.principal_x
    if args.principal_y is not None:
        config.reconstruction.principal_point_y = args.principal_y
    if args.host:
        config.gui.host = args.host
    if args.port:
        config.gui.port = args.port
    if args.loop:
        config.video.loop = True
    validate_config(config)

    pipeline_holder: dict[str, RealtimePipeline] = {}
    dashboard = None
    scene = None
    if not args.headless:
        dashboard = Dashboard(
            config.gui,
            lambda command, value: pipeline_holder["pipeline"].handle_command(command, value),
            reconstruction_only=config.mode == "reconstruction",
            people_overlay=config.people_overlay,
        )
        scene = (
            ReconstructionScene3D(dashboard.server, config.gui)
            if config.mode == "reconstruction"
            else Scene3D(dashboard.server, config.gui)
        )
    output_dir = Path(args.output_dir or f"sessions/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    session_logger = None if args.no_log else SessionLogger(output_dir)
    recorder = VideoRecorder(output_dir / "annotated.mp4") if args.record else None
    pipeline = RealtimePipeline(
        config,
        dashboard=dashboard,
        scene=scene,
        session_logger=session_logger,
        video_recorder=recorder,
    )
    pipeline_holder["pipeline"] = pipeline
    shutdown = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    pipeline.start_workers()
    try:
        if args.source is not None:
            source = int(args.source) if args.source.isdigit() else args.source
            pipeline.start_source(source, max_frames=args.max_frames)
        elif args.auto_webcam:
            try:
                pipeline.start_source("auto", max_frames=args.max_frames)
            except (CameraDetectionError, RuntimeError) as exc:
                if args.headless:
                    raise
                logging.info("No readable webcam detected; waiting for a GUI upload or source selection: %s", exc)
                if dashboard is not None:
                    dashboard.update_camera_status(f"Webcam: **not detected** — {exc}")
        elif args.headless:
            raise ValueError("--source is required in headless mode when --no-auto-webcam is used")

        deadline = time.perf_counter() + args.duration if args.duration else None
        while not shutdown.is_set():
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if args.exit_on_end and pipeline.source_done:
                break
            if args.headless and pipeline.source_done:
                break
            shutdown.wait(0.1)
    finally:
        final_cloud = pipeline.gui_state.read().pointcloud
        pipeline.close()
        if scene is not None:
            scene.close()
        if dashboard is not None:
            dashboard.close()
        if args.export_ply and final_cloud is not None:
            export_ply(final_cloud, output_dir / "final_pointcloud.ply")
    snapshot = pipeline.gui_state.read()
    perf = snapshot.performance
    logging.info(
        "Final actual rates: input=%.2f display=%.2f segmentation=%.2f 3D=%.2f safety=%.2f FPS, latency avg/p95=%.1f/%.1f ms, dropped=%d",
        perf.input_fps,
        perf.display_fps,
        perf.segmentation_fps,
        perf.reconstruction_fps,
        perf.safety_fps,
        perf.average_latency_ms,
        perf.p95_latency_ms,
        perf.dropped_frames,
    )
    if pipeline.errors:
        logging.warning("Runtime fallbacks/errors: %s", pipeline.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
