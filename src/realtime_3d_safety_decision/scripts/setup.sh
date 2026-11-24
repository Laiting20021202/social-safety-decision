#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"

# Reuse a working host PyTorch/CUDA installation when one is available.  This
# avoids downloading another multi-gigabyte CUDA runtime into the project venv.
VENV_ARGS=()
if python3 -c 'import torch' >/dev/null 2>&1; then
  VENV_ARGS+=(--system-site-packages)
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  python3 -m venv "${VENV_ARGS[@]}" "${VENV_PATH}"
elif (( ${#VENV_ARGS[@]} > 0 )) && ! "${VENV_PATH}/bin/python" -c 'import torch' >/dev/null 2>&1; then
  echo "Existing venv cannot see the host CUDA/PyTorch installation; rebuilding it."
  python3 -m venv --clear "${VENV_ARGS[@]}" "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip 'setuptools>=68,<80' wheel
"${VENV_PATH}/bin/python" -m pip install --no-cache-dir -r "${ROOT_DIR}/requirements.txt"
"${VENV_PATH}/bin/python" - <<'PY'
import torch

print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
PY
echo "Installed. Activate with: source ${VENV_PATH}/bin/activate"
