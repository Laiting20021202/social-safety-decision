#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SOCIAL_ROOT="${SOCIAL_SAFETY_ROOT:-${REPOSITORY_ROOT}}"
RESULTS_DIR="${OPENARM_DYNAMIC_TEST_RESULTS_DIR:-${PROJECT_ROOT}/results}"
TEST_TIMEOUT="${OPENARM_DYNAMIC_TEST_TIMEOUT:-150}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OPENARM_DYNAMIC_TEST_OUTPUT:-${RESULTS_DIR}/dynamic_dual_target_${STAMP}.json}"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[FAIL] ROS 2 Humble is not installed." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/install/setup.bash" ]]; then
  echo "[FAIL] OpenArm ROS workspace is not built." >&2
  echo "[INFO] Start the demo first: ${PROJECT_ROOT}/scripts/start_gazebo_demo.sh" >&2
  exit 1
fi
if [[ ! -f "${SOCIAL_ROOT}/scripts/ros2_lan_env.sh" ]]; then
  echo "[FAIL] ROS LAN environment not found under ${SOCIAL_ROOT}." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "${PROJECT_ROOT}/install/setup.bash"
source "${SOCIAL_ROOT}/scripts/ros2_lan_env.sh"
set -u

if ! timeout 6 ros2 topic echo --once /clock >/dev/null 2>&1; then
  echo "[FAIL] Gazebo demo is not ready: /clock has no data." >&2
  echo "[INFO] Run ${PROJECT_ROOT}/scripts/start_gazebo_demo.sh and wait for OPENARM GAZEBO DEMO READY." >&2
  exit 1
fi
if ! timeout 6 ros2 topic echo --once /openarm/dynamic_avoidance/status >/dev/null 2>&1; then
  echo "[FAIL] Dynamic avoidance node is not ready." >&2
  echo "[INFO] Inspect ${PROJECT_ROOT}/.demo/logs/gazebo_demo.log" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
echo "[INFO] Running perception-only moving-hand avoidance test..."
echo "[INFO] Do not operate Gazebo or the web GUI until this test finishes."

stop_hand() {
  timeout 2s ros2 topic pub -r 10 /sim/hand/command std_msgs/msg/String \
    "{data: 'auto_sweep:off'}" >/dev/null 2>&1 || true
  timeout 2s ros2 topic pub -r 10 /sim/hand/command std_msgs/msg/String \
    "{data: 'pause'}" >/dev/null 2>&1 || true
}
trap stop_hand EXIT

ros2 service call /openarm/safety/reset_estop std_srvs/srv/Trigger '{}' >/dev/null 2>&1 || true
timeout 2s ros2 topic pub -r 10 /openarm/obstacle_source std_msgs/msg/String \
  "{data: 'perception'}" >/dev/null 2>&1 || true
timeout 2s ros2 topic pub -r 10 /openarm/planner/mode std_msgs/msg/String \
  "{data: 'dynamic'}" >/dev/null 2>&1 || true
timeout 2s ros2 topic pub -r 10 /sim/hand/command std_msgs/msg/String \
  "{data: 'speed:0.01'}" >/dev/null 2>&1 || true
timeout 2s ros2 topic pub -r 10 /sim/hand/command std_msgs/msg/String \
  "{data: 'resume'}" >/dev/null 2>&1 || true
timeout 2s ros2 topic pub -r 10 /sim/hand/command std_msgs/msg/String \
  "{data: 'auto_sweep:off'}" >/dev/null 2>&1 || true

set +e
python3 "${PROJECT_ROOT}/scripts/validate_dual_target_motion.py" \
  --timeout "${TEST_TIMEOUT}" \
  --tolerance 0.055 \
  --post-target-avoidance \
  --output "${OUTPUT}" \
  "$@"
STATUS=$?
set -e
stop_hand
trap - EXIT

if [[ ${STATUS} -eq 0 ]]; then
  echo "[PASS] Dynamic avoidance test"
else
  echo "[FAIL] Dynamic avoidance test" >&2
fi
echo "[INFO] Result: ${OUTPUT}"
exit "${STATUS}"
