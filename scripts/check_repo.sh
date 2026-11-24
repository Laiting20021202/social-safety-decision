#!/usr/bin/env bash
# Offline source checks only: no ROS nodes, GUI, simulator or control commands.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERCEPTION_DIR="$ROOT_DIR/src/realtime_3d_safety_decision"
PYTHON="${REPO_CHECK_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ -z "${REPO_CHECK_PYTHON:-}" && ! -x "$PYTHON" ]]; then
  PYTHON="$PERCEPTION_DIR/.venv/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python: $PYTHON. See docs/DOCKER.md for offline test setup." >&2
  exit 1
fi

cd "$ROOT_DIR"
git diff --check
while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(git ls-files --cached --others --exclude-standard -z -- '*.sh')

"$PYTHON" -m compileall -q src scripts docker
for package_dir in "$PERCEPTION_DIR" "$ROOT_DIR/src/openarm_sim"; do
  (
    cd "$package_dir"
    "$PYTHON" -m pytest -q
  )
done
echo "Offline checks passed. Isaac runtime and arm avoidance still require separate validation."
