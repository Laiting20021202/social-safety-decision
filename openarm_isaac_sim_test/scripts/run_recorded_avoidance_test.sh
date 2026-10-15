#!/usr/bin/env bash
# Existing perception/controller trial, with isolated cameras/contact evidence.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
OUTPUT="${1:-$ROOT/results/gazebo_restore/trial_$(date +%Y%m%d_%H%M%S)}"
shift || true
mkdir -p "$OUTPUT"
set +u
source /opt/ros/humble/setup.bash
source "$PROJECT_ROOT/install/setup.bash"
export ROS_LOCAL_SIM=1 KOCH_ROS_DOMAIN_ID=79 OPENBLAS_NUM_THREADS=1
source "$ROOT/scripts/ros2_lan_env.sh"
set -u
python3 "$PROJECT_ROOT/scripts/spawn_recording_camera.py"
python3 "$PROJECT_ROOT/scripts/record_gazebo_trial.py" --seconds 360 --output "$OUTPUT/uncut.mp4" >"$OUTPUT/recording.log" 2>&1 &
VIDEO_PID=$!
python3 "$PROJECT_ROOT/scripts/evaluate_gazebo_contacts.py" --seconds 360 --output "$OUTPUT/contacts.json" >"$OUTPUT/contacts.log" 2>&1 &
CONTACT_PID=$!
finish() {
  kill -INT "$VIDEO_PID" "$CONTACT_PID" 2>/dev/null || true
  wait "$VIDEO_PID" || true
  wait "$CONTACT_PID" || true
}
trap finish EXIT
for _ in $(seq 1 60); do
  if rg -q '^RECORDING' "$OUTPUT/recording.log"; then break; fi
  if ! kill -0 "$VIDEO_PID" 2>/dev/null; then cat "$OUTPUT/recording.log"; exit 1; fi
  sleep .25
done
if ! rg -q '^RECORDING' "$OUTPUT/recording.log"; then echo 'Recording failed to start'; exit 1; fi
set +e
OPENARM_DYNAMIC_TEST_OUTPUT="$OUTPUT/task.json" OPENARM_DYNAMIC_TEST_TIMEOUT=110 \
  bash "$PROJECT_ROOT/scripts/run_dynamic_avoidance_test.sh" "$@" >"$OUTPUT/task.log" 2>&1
TASK_STATUS=$?
set -e
finish
trap - EXIT
python3 - "$OUTPUT" "$TASK_STATUS" <<'PY'
import json,sys,subprocess,hashlib
from pathlib import Path
p=Path(sys.argv[1]); task_ok=int(sys.argv[2])==0
contacts=json.loads((p/'contacts.json').read_text()) if (p/'contacts.json').exists() else {}
video=json.loads((p/'uncut.json').read_text()) if (p/'uncut.json').exists() else {}
report=dict(task_passed=task_ok,contact_check_passed=contacts.get('passed',False),video_encoded=video.get('encoder_exit')==0,
 controller='geometric + visibility memory; no trained action policy',perception='RGB-D baseline (simulated depth); not RGB-only',
 base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_modified=True,
 task_configuration_sha256=hashlib.sha256(Path('openarm_isaac_sim_test/config/sorting_task.yaml').read_bytes()).hexdigest(),seed=20260817,
 passed=task_ok and contacts.get('passed',False) and video.get('encoder_exit')==0)
(p/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
PY
