#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VIDEO_PATH [extra app.py arguments]" >&2
  exit 2
fi
exec python3 "${ROOT_DIR}/app.py" --source "$1" --profile agx --device cuda "${@:2}"
