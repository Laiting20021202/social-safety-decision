#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WITH_ST4RTRACK=0
if [[ "${1:-}" == "--st4rtrack" ]]; then
  WITH_ST4RTRACK=1
fi

cd "${ROOT_DIR}"
"${PYTHON}" - <<'PY'
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

YOLO("yolo11n-seg.pt")
name = "depth-anything/Depth-Anything-V2-Small-hf"
AutoImageProcessor.from_pretrained(name)
AutoModelForDepthEstimation.from_pretrained(name)
print("YOLO11n-seg and Depth Anything V2 Small are cached.")
PY

if [[ "${WITH_ST4RTRACK}" -eq 1 ]]; then
  mkdir -p "${ROOT_DIR}/third_party"
  if [[ ! -d "${ROOT_DIR}/third_party/St4RTrack/.git" ]]; then
    git clone --depth 1 https://github.com/HavenFeng/St4RTrack.git "${ROOT_DIR}/third_party/St4RTrack"
  fi
  echo "St4RTrack code installed. Its checkpoint downloads on first st4rtrack/hybrid load from yupengchengg147/St4RTrack."
  echo "Upstream St4RTrack code/checkpoints are non-commercial research assets."
fi
