#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

SOURCE_ARGS=()
if [[ $# -gt 0 && "${1}" != --* ]]; then
  SOURCE_ARGS=(--source "$1")
  shift
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" app.py --profile st4rtrack_viewer --device cuda "${SOURCE_ARGS[@]}" "$@"
