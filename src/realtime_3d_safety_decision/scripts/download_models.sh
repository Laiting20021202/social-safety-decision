#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
WITH_ST4RTRACK=0
WITH_VIDEO_DEPTH=0
WITH_SAFETY_MODELS=1
if [[ "${1:-}" == "--st4rtrack" ]]; then
  WITH_ST4RTRACK=1
elif [[ "${1:-}" == "--video-depth" ]]; then
  WITH_VIDEO_DEPTH=1
  WITH_SAFETY_MODELS=0
elif [[ "${1:-}" == "--viewer" ]]; then
  WITH_ST4RTRACK=1
  WITH_VIDEO_DEPTH=1
  WITH_SAFETY_MODELS=0
fi

if [[ "${WITH_VIDEO_DEPTH}" -eq 1 ]]; then
  mkdir -p "${ROOT_DIR}/third_party"
  if [[ ! -d "${ROOT_DIR}/third_party/Video-Depth-Anything/.git" ]]; then
    git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git \
      "${ROOT_DIR}/third_party/Video-Depth-Anything"
  fi
  "${PYTHON}" - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="depth-anything/Metric-Video-Depth-Anything-Small",
    filename="metric_video_depth_anything_vits.pth",
)
print("Metric Video Depth Anything Small is cached.")
PY
fi

cd "${ROOT_DIR}"
if [[ "${WITH_SAFETY_MODELS}" -eq 1 ]]; then
  "${PYTHON}" - <<'PY'
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

YOLO("yolo11n-seg.pt")
name = "depth-anything/Depth-Anything-V2-Small-hf"
AutoImageProcessor.from_pretrained(name)
AutoModelForDepthEstimation.from_pretrained(name)
print("YOLO11n-seg and Depth Anything V2 Small are cached.")
PY
fi

if [[ "${WITH_ST4RTRACK}" -eq 1 ]]; then
  mkdir -p "${ROOT_DIR}/third_party"
  if [[ ! -d "${ROOT_DIR}/third_party/St4RTrack/.git" ]]; then
    git clone --depth 1 https://github.com/HavenFeng/St4RTrack.git "${ROOT_DIR}/third_party/St4RTrack"
  fi
  HF_XET_CHUNK_CACHE_SIZE_BYTES=0 "${PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yupengchengg147/St4RTrack",
    allow_patterns=["config.json", "model.safetensors"],
)
print("St4RTrack sequence checkpoint is cached.")
PY
  echo "St4RTrack code and sequence checkpoint installed."
  echo "Upstream St4RTrack code/checkpoints are non-commercial research assets."
fi
