#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
echo "Installed. Activate with: source ${VENV_PATH}/bin/activate"
