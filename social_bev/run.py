from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from social_bev.config import load_config
from social_bev.frame_source import FrameSource
from social_bev.output_writer import OutputWriter
from social_bev.pipeline import SocialNavigationPipeline
from social_bev.utils import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only RGB social navigation BEV pipeline")
    parser.add_argument("--input", required=True, help="Video, image, image directory, SCAND sample directory, or webcam index")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML runtime config")
    parser.add_argument("--calibration", default=None, help="Calibration YAML; omitted means NON-METRIC BEV")
    parser.add_argument("--output", default="outputs/result.mp4", help="Output annotated 2x2 video")
    parser.add_argument("--jsonl", default=None, help="Output JSONL path")
    parser.add_argument("--occupancy-dir", default=None, help="Output occupancy directory")
    parser.add_argument("--device", default="cpu", help="Only cpu is supported")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames")
    parser.add_argument("--stride", type=int, default=1, help="Input frame stride")
    parser.add_argument("--display", action="store_true", help="Show processing window if a display is available")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    config.setdefault("runtime", {})["device"] = "cpu"
    if str(args.device).lower() != "cpu":
        LOGGER.warning("Ignoring --device %s because this project supports CPU execution only", args.device)
    source = FrameSource(args.input, stride=args.stride)
    pipeline = SocialNavigationPipeline(config=config, calibration_path=args.calibration)
    jsonl_path = args.jsonl or config.get("runtime", {}).get("output_jsonl", "outputs/results.jsonl")
    occupancy_dir = args.occupancy_dir or config.get("runtime", {}).get("occupancy_dir", "outputs/occupancy")
    display = bool(args.display) and _display_available() and _opencv_highgui_available()
    if args.display and not display:
        LOGGER.info("Display disabled because no GUI display or OpenCV HighGUI backend is available")

    processed = 0
    writer: OutputWriter | None = None
    try:
        iterator = iter(source)
        progress = tqdm(iterator, desc="social-bev", unit="frame")
        for frame in progress:
            if writer is None:
                writer = OutputWriter(
                    video_path=args.output,
                    jsonl_path=jsonl_path,
                    occupancy_dir=occupancy_dir,
                    fps=source.fps,
                )
            result = pipeline.process_frame(frame.image, frame.timestamp)
            writer.write(result)
            processed += 1
            progress.set_postfix(fps=f"{result.fps:.2f}", avg=f"{result.average_fps:.2f}")
            if display:
                cv2.imshow("rgb social navigation bev", result.visualization)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.close()
        if display:
            cv2.destroyAllWindows()

    if processed == 0:
        raise RuntimeError(f"No frames were processed from input: {args.input}")
    LOGGER.info("Processed %d frames", processed)
    LOGGER.info("Video: %s", Path(args.output))
    LOGGER.info("JSONL: %s", Path(jsonl_path))
    LOGGER.info("Occupancy: %s", Path(occupancy_dir))


def _display_available() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _opencv_highgui_available() -> bool:
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        return False
    for line in build_info.splitlines():
        if "GUI:" in line:
            return "NONE" not in line.upper()
    return True


if __name__ == "__main__":
    main()
