from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from social_bev.config import load_classes, load_config
from social_bev.frame_source import FrameSource
from social_bev.output_writer import OutputWriter
from social_bev.pipeline import SocialNavigationPipeline


st.set_page_config(page_title="RGB Social Navigation BEV", layout="wide")
st.title("RGB Social Navigation BEV")


@st.cache_data
def cached_defaults() -> tuple[dict, dict]:
    return load_config("configs/default.yaml"), load_classes("configs/classes.yaml")


config, class_config = cached_defaults()
runtime = config.setdefault("runtime", {})
segmentation = config.setdefault("segmentation", {})
detection = config.setdefault("detection", {})
social_zone = config.setdefault("social_zone", {})

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    local_video = st.text_input("Local video path", "data/input.mp4")
    use_scand = st.checkbox("Use SCAND sample", value=False)
    scand_path = st.text_input("SCAND image directory", "data/scand_sample/images")
    calibration_path = st.text_input("Calibration YAML", "")

    st.header("Runtime")
    detection["confidence"] = st.slider("Detection confidence", 0.05, 0.90, float(detection.get("confidence", 0.35)), 0.05)
    stride = st.number_input("Frame stride", min_value=1, max_value=30, value=1, step=1)
    max_frames = st.number_input("Max frames", min_value=1, max_value=10000, value=200, step=10)

    st.header("Walkable Classes")
    available_classes = sorted(set(class_config["walkable_classes"] + class_config["optional_walkable_classes"] + ["grass", "earth", "field", "rug"]))
    class_config["walkable_classes"] = st.multiselect(
        "Walkable",
        available_classes,
        default=class_config["walkable_classes"],
    )

    st.header("Social Zone")
    social_zone["static_radius_m"] = st.slider("Static radius m", 0.2, 2.0, float(social_zone.get("static_radius_m", 0.8)), 0.1)
    social_zone["front_radius_m"] = st.slider("Front radius m", 0.2, 3.0, float(social_zone.get("front_radius_m", 1.2)), 0.1)
    social_zone["rear_radius_m"] = st.slider("Rear radius m", 0.2, 2.0, float(social_zone.get("rear_radius_m", 0.6)), 0.1)

    st.header("Output")
    output_video = st.text_input("Output video", "outputs/streamlit_result.mp4")
    output_jsonl = st.text_input("Output JSONL", "outputs/streamlit_results.jsonl")
    output_occupancy = st.text_input("Output occupancy dir", "outputs/streamlit_occupancy")
    run_button = st.button("Run")

progress = st.progress(0.0)
status = st.empty()
preview = st.empty()
stats = st.empty()

if run_button:
    source_path: str | int
    temp_path: Path | None = None
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)
        with temp_file:
            while True:
                chunk = uploaded.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
        source_path = str(temp_path)
    elif use_scand:
        source_path = scand_path
    else:
        source_path = local_video

    runtime["display"] = False
    runtime["device"] = "cpu"
    config["detection"] = detection
    config["social_zone"] = social_zone
    source = FrameSource(source_path, stride=int(stride))
    pipeline = SocialNavigationPipeline(
        config=config,
        calibration_path=calibration_path or None,
        class_config=class_config,
    )
    writer: OutputWriter | None = None
    processed = 0
    fps_values: list[float] = []
    try:
        for frame in source:
            if writer is None:
                writer = OutputWriter(output_video, output_jsonl, output_occupancy, fps=source.fps)
            result = pipeline.process_frame(frame.image, frame.timestamp)
            writer.write(result)
            processed += 1
            fps_values.append(result.fps)
            progress.progress(min(1.0, processed / float(max_frames)))
            status.write(f"Processed {processed} frames")
            preview.image(cv2.cvtColor(result.visualization, cv2.COLOR_BGR2RGB), channels="RGB")
            stats.json(
                {
                    "current_fps": round(result.fps, 3),
                    "average_fps": round(result.average_fps, 3),
                    "processing_ms": result.processing_ms,
                    "metric_bev": result.bev.metric_bev,
                    "video": output_video,
                    "jsonl": output_jsonl,
                }
            )
            if processed >= int(max_frames):
                break
    finally:
        if writer is not None:
            writer.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    st.success(f"Finished {processed} frames. Output video: {output_video}")

