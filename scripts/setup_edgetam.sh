#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGETAM_COMMIT="7711e012a30a2402c4eaab637bdb00a521302c91"
EDGETAM_REPOSITORY_URL="https://github.com/facebookresearch/EdgeTAM.git"
EDGETAM_ROOT="${EDGETAM_ROOT:-${ROOT_DIR}/third_party/EdgeTAM}"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
PYTORCH29_PATCH="${ROOT_DIR}/patches/edgetam_pytorch29_grouped_objects.patch"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  # ROS 2 Humble's Python packages live under /opt/ros rather than PyPI.
  # Keeping system site packages visible lets the activated environment use
  # rclpy after /opt/ros/humble/setup.bash has been sourced.
  python3 -m venv --system-site-packages "${VENV_PATH}"
fi
PYTHON="${EDGETAM_PYTHON:-${VENV_PATH}/bin/python}"

"${PYTHON}" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "Refusing to install EdgeTAM into the system Python. "
        "Set VENV_PATH or EDGETAM_PYTHON to a virtual-environment interpreter."
    )
PY

if [[ -e "${EDGETAM_ROOT}" && ! -d "${EDGETAM_ROOT}/.git" ]]; then
  echo "EdgeTAM target exists but is not a Git checkout: ${EDGETAM_ROOT}" >&2
  exit 1
fi

NEW_CLONE=0
if [[ ! -d "${EDGETAM_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${EDGETAM_ROOT}")"
  git clone --filter=blob:none --no-checkout \
    "${EDGETAM_REPOSITORY_URL}" "${EDGETAM_ROOT}"
  NEW_CLONE=1
fi

if [[ "${NEW_CLONE}" == "1" ]]; then
  # The official repository includes checkpoints, notebooks, and demo videos.
  # Production only needs the Python package and root packaging metadata.
  # Cone mode includes root files plus the selected source directory.
  git -C "${EDGETAM_ROOT}" sparse-checkout init --cone
  git -C "${EDGETAM_ROOT}" sparse-checkout set sam2
else
  if ! git -C "${EDGETAM_ROOT}" diff --quiet ||
     ! git -C "${EDGETAM_ROOT}" diff --cached --quiet; then
    # The project carries one audited compatibility patch. Accept an
    # idempotent setup rerun only when that exact patch is the whole diff.
    if ! cmp -s \
      <(git -C "${EDGETAM_ROOT}" diff -- sam2/modeling/perceiver.py) \
      "${PYTORCH29_PATCH}" ||
      [[ "$(git -C "${EDGETAM_ROOT}" diff --name-only | wc -l)" -ne 1 ]] ||
      ! git -C "${EDGETAM_ROOT}" diff --cached --quiet; then
      echo "EdgeTAM checkout has unexpected local changes; refusing to overwrite them: ${EDGETAM_ROOT}" >&2
      exit 1
    fi
  fi
fi

CURRENT_COMMIT="$(git -C "${EDGETAM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${CURRENT_COMMIT}" != "${EDGETAM_COMMIT}" ]]; then
  git -C "${EDGETAM_ROOT}" fetch --depth 1 origin "${EDGETAM_COMMIT}"
fi
if [[ "${NEW_CLONE}" == "1" || "${CURRENT_COMMIT}" != "${EDGETAM_COMMIT}" ]]; then
  git -C "${EDGETAM_ROOT}" checkout --detach "${EDGETAM_COMMIT}"
fi

# EdgeTAM commit 7711e0 uses view() on an expanded zero-stride latent tensor.
# PyTorch 2.9 rejects that for B>1, which otherwise forces the wrapper to run
# one full predictor per object. Keep the pinned upstream commit and apply the
# minimal reshape compatibility patch reproducibly.
if git -C "${EDGETAM_ROOT}" apply --reverse --check "${PYTORCH29_PATCH}"; then
  echo "EdgeTAM PyTorch 2.9 grouped-object patch already applied"
elif git -C "${EDGETAM_ROOT}" apply --check "${PYTORCH29_PATCH}"; then
  git -C "${EDGETAM_ROOT}" apply "${PYTORCH29_PATCH}"
  echo "Applied EdgeTAM PyTorch 2.9 grouped-object patch"
else
  echo "EdgeTAM compatibility patch does not match the pinned source" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    import torch
    import torchvision
except ImportError as exc:
    raise SystemExit(
        "PyTorch and torchvision must be installed in the project virtual environment first. "
        "Use the official PyTorch selector for the host CUDA version."
    ) from exc

def major_minor(value: str) -> tuple[int, int]:
    public = value.split("+", 1)[0]
    major, minor, *_ = public.split(".")
    return int(major), int(minor)

if major_minor(torch.__version__) < (2, 3):
    raise SystemExit(f"EdgeTAM requires torch>=2.3.1; found {torch.__version__}")
if major_minor(torchvision.__version__) < (0, 18):
    raise SystemExit(
        f"EdgeTAM requires torchvision>=0.18.1; found {torchvision.__version__}"
    )
print(f"Using torch {torch.__version__} and torchvision {torchvision.__version__}")
PY

"${PYTHON}" -m pip install \
  "timm==1.0.15" \
  "eva-decord>=0.6.1" \
  "hydra-core>=1.3.2,<1.4" \
  "iopath>=0.1.10,<0.2"

if [[ -z "${SAM2_BUILD_CUDA:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    SAM2_BUILD_CUDA=1
  else
    SAM2_BUILD_CUDA=0
  fi
fi
export SAM2_BUILD_CUDA
# EdgeTAM lists torch as a PEP 517 build dependency. A normal isolated build
# downloads another multi-gigabyte torch wheel even though the validated
# project environment already provides it. Reuse that exact environment and
# avoid re-resolving the runtime stack we checked above.
"${PYTHON}" -m pip install \
  --no-build-isolation \
  --no-deps \
  --editable "${EDGETAM_ROOT}"

INSTALLED_COMMIT="$(git -C "${EDGETAM_ROOT}" rev-parse HEAD)"
if [[ "${INSTALLED_COMMIT}" != "${EDGETAM_COMMIT}" ]]; then
  echo "Pinned EdgeTAM commit verification failed: ${INSTALLED_COMMIT}" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import torch
import torchvision
import timm
from sam2.build_sam import build_sam2_video_predictor

print(f"EdgeTAM import OK; torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}; timm={timm.__version__}")
print(f"CUDA available={torch.cuda.is_available()}; CUDA runtime={torch.version.cuda}")
print(f"Official builder={build_sam2_video_predictor.__module__}")
PY

echo "EdgeTAM source: ${EDGETAM_ROOT}"
echo "Pinned commit: ${EDGETAM_COMMIT}"
echo "Python environment: ${PYTHON}"
echo "Download the checkpoint with scripts/download_edgetam_checkpoint.sh"
