#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLAM_DIR="${ROOT_DIR}/third_party/MASt3R-SLAM"
VENV_DIR="${ROOT_DIR}/.venv-mast3r-slam"
BASE_PYTHON="${MAST3R_SLAM_BASE_PYTHON:-python3}"

available_kb="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {print $4}')"
needs_initial_checkpoint_download=false
for checkpoint_name in \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
do
  if [[ ! -s "${SLAM_DIR}/checkpoints/${checkpoint_name}" ]]; then
    needs_initial_checkpoint_download=true
    break
  fi
done
if [[ "${needs_initial_checkpoint_download}" == true ]] \
  && { [[ -z "${available_kb}" ]] || [[ "${available_kb}" -lt 8388608 ]]; }
then
  echo "MASt3R-SLAM setup needs at least 8 GiB of free disk space." >&2
  echo "Available: $(( ${available_kb:-0} / 1024 / 1024 )) GiB" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/third_party"
if [[ ! -d "${SLAM_DIR}/.git" ]]; then
  git clone https://github.com/rmurai0610/MASt3R-SLAM.git "${SLAM_DIR}"
fi
# The isolated headless worker needs Eigen for the CUDA extension. Upstream's
# in3d/pyimgui submodule is used only by its ModernGL window and requires Git
# LFS, so deliberately do not install that unrelated visualization stack.
git -C "${SLAM_DIR}" submodule update --init thirdparty/eigen

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  # Reuse the host CUDA-enabled PyTorch wheel, but isolate MASt3R's dust3r
  # package from the main application's St4RTrack environment.
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" -m pip install --upgrade pip 'setuptools>=68,<80' wheel ninja packaging
"${PYTHON}" -m pip install \
  'numpy==1.26.4' einops natsort plyfile roma pyaml faiss-cpu huggingface-hub
"${PYTHON}" -m pip install --no-deps -e "${SLAM_DIR}/thirdparty/mast3r/asmk"
"${PYTHON}" -m pip install --no-deps -e "${SLAM_DIR}/thirdparty/mast3r"
"${PYTHON}" -m pip install --no-build-isolation \
  'lietorch @ git+https://github.com/princeton-vl/lietorch.git'

CUDA_ROOT="${CUDA_HOME:-}"
if [[ -z "${CUDA_ROOT}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.2; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      CUDA_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CUDA_ROOT}" || ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "CUDA toolkit/nvcc was not found; it is required to build MASt3R-SLAM." >&2
  exit 1
fi
CUDA_HOME="${CUDA_ROOT}" PATH="${CUDA_ROOT}/bin:${PATH}" \
  bash -c '
    patch_file="$1"
    slam_dir="$2"
    if git -C "${slam_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
      git -C "${slam_dir}" apply "${patch_file}"
    elif git -C "${slam_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
      echo "PyTorch 2.9 compatibility patch is already applied."
    else
      echo "Could not apply the MASt3R-SLAM PyTorch compatibility patch." >&2
      exit 1
    fi
  ' bash "${ROOT_DIR}/patches/mast3r_slam_pytorch29.patch" "${SLAM_DIR}"
CUDA_HOME="${CUDA_ROOT}" PATH="${CUDA_ROOT}/bin:${PATH}" \
  "${PYTHON}" -m pip install --no-build-isolation --no-deps -e "${SLAM_DIR}"

mkdir -p "${SLAM_DIR}/checkpoints"
checkpoint_base="https://download.europe.naverlabs.com/ComputerVision/MASt3R"
for filename in \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
do
  url="${checkpoint_base}/${filename}"
  destination="${SLAM_DIR}/checkpoints/${filename}"
  remote_size="$(
    curl --fail --silent --show-error --location --head "${url}" \
      | tr -d '\r' \
      | awk 'tolower($1) == "content-length:" { size=$2 } END { print size }'
  )"
  local_size="$(stat --format='%s' "${destination}" 2>/dev/null || echo 0)"
  if [[ -z "${remote_size}" || ! "${remote_size}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the size of ${filename}." >&2
    exit 1
  fi
  if [[ "${local_size}" != "${remote_size}" ]]; then
    partial="${destination}.part"
    if [[ -s "${destination}" && ! -e "${partial}" ]]; then
      mv "${destination}" "${partial}"
    fi
    curl --fail --location --continue-at - \
      "${url}" --output "${partial}"
    downloaded_size="$(stat --format='%s' "${partial}")"
    if [[ "${downloaded_size}" != "${remote_size}" ]]; then
      echo "Incomplete checkpoint ${filename}: ${downloaded_size}/${remote_size} bytes." >&2
      exit 1
    fi
    mv "${partial}" "${destination}"
  fi
done

"${PYTHON}" - <<'PY'
import torch
import lietorch
import mast3r_slam_backends

assert torch.cuda.is_available(), "CUDA is not available in the MASt3R-SLAM environment"
print(f"MASt3R-SLAM environment ready: torch={torch.__version__}, GPU={torch.cuda.get_device_name()}")
PY

echo "Installed official MASt3R-SLAM in ${SLAM_DIR}"
echo "The MASt3R checkpoints have separate research/non-commercial license terms; review upstream LICENSE.md and CHECKPOINTS_NOTICE."
