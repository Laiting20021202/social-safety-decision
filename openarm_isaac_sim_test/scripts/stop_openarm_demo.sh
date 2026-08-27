#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${OPENARM_DEMO_STATE_DIR:-${PROJECT_ROOT}/.demo}"
LAUNCH_PID_FILE="${STATE_DIR}/launch.pid"
QUIET=false
[[ "${1:-}" == "--quiet" ]] && QUIET=true

# During an in-place update we may temporarily run these nodes as
# tracked user services so Gazebo poses are preserved. Always remove them
# before a normal one-click restart/stop to prevent duplicate ROS node names.
systemctl --user stop openarm-pose-goal-hotfix.service 2>/dev/null || true
systemctl --user stop openarm-dynamic-avoidance-hotfix.service 2>/dev/null || true
systemctl --user stop openarm-safety-hotfix.service 2>/dev/null || true
systemctl --user stop openarm-resampler-hotfix.service 2>/dev/null || true
systemctl --user stop openarm-hand-hotfix.service 2>/dev/null || true

if [[ -f "${LAUNCH_PID_FILE}" ]]; then
  LAUNCH_PID="$(tr -cd '0-9' <"${LAUNCH_PID_FILE}")"
  if [[ -n "${LAUNCH_PID}" && "${LAUNCH_PID}" -gt 1 ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null || kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "${LAUNCH_PID}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      kill -KILL -- "-${LAUNCH_PID}" 2>/dev/null || kill -KILL "${LAUNCH_PID}" 2>/dev/null || true
    fi
  fi
  rm -f "${LAUNCH_PID_FILE}"
fi

# launch_ros starts Gazebo's server/client in their own process groups. If the
# launch parent is stopped first they are re-parented and keep simulating, so
# also stop only processes whose command line contains this demo's exact world
# or RViz config path.
WORLD_PATH="${PROJECT_ROOT}/install/openarm_sim_bringup/share/openarm_sim_bringup/gazebo/worlds/openarm_sorting.world"
RVIZ_PATH="${PROJECT_ROOT}/install/openarm_sim_bringup/share/openarm_sim_bringup/rviz/openarm_gazebo_phase1.rviz"
mapfile -t DEMO_PIDS < <(pgrep -f -- "${WORLD_PATH}|${RVIZ_PATH}" || true)
if (( ${#DEMO_PIDS[@]} )); then
  kill -TERM "${DEMO_PIDS[@]}" 2>/dev/null || true
  survivors=("${DEMO_PIDS[@]}")
  for _ in $(seq 1 25); do
    survivors=()
    for pid in "${DEMO_PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null && survivors+=("${pid}")
    done
    (( ${#survivors[@]} == 0 )) && break
    sleep 0.2
  done
  if (( ${#survivors[@]} )); then
    kill -KILL "${survivors[@]}" 2>/dev/null || true
  fi
fi

systemctl --user stop realtime-safety-3d.service 2>/dev/null || true
${QUIET} || echo "OPENARM DEMO STOPPED"
