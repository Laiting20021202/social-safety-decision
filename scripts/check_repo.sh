#!/usr/bin/env bash
# Offline source checks only: no ROS nodes, GUI, simulator or control commands.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_CHECK_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python: $PYTHON. Run scripts/setup.sh and install .[test] first." >&2
  exit 1
fi

cd "$ROOT_DIR"
git diff --check
while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(git ls-files --cached --others --exclude-standard -z -- '*.sh')

"$PYTHON" -m compileall -q app.py realtime_safety scripts tests \
  openarm_isaac_sim_test/openarm_sim openarm_isaac_sim_test/scripts \
  openarm_isaac_sim_test/launch
"$PYTHON" -m pytest -q tests
(
  cd openarm_isaac_sim_test
  "$PYTHON" -m pytest -q
)
echo "Offline checks passed. Isaac runtime and arm avoidance still require separate validation."
