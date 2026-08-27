#!/usr/bin/env bash
set -euo pipefail

if [[ "${OPENARM_SIM_BACKEND:-gazebo}" == "gazebo" ]]; then
  exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_gazebo_demo.sh" "${@}"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SOCIAL_ROOT="${SOCIAL_SAFETY_ROOT:-${REPOSITORY_ROOT}}"
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-${ISAAC_SIM_ROOT:-/home/david/isaacsim}/python.sh}"
STATE_DIR="${OPENARM_DEMO_STATE_DIR:-${PROJECT_ROOT}/.demo}"
LOG_DIR="${STATE_DIR}/logs"
LAUNCH_PID_FILE="${STATE_DIR}/launch.pid"
READY_TIMEOUT="${OPENARM_DEMO_READY_TIMEOUT:-240}"
GUI_URL="${OPENARM_GUI_URL:-http://192.168.0.234:8080/}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=0

mkdir -p "${LOG_DIR}"

if [[ ! -x "${ISAAC_SIM_PYTHON}" ]]; then
  echo "[FAIL] Isaac Sim Python: ${ISAAC_SIM_PYTHON}" >&2
  echo "[INFO] Set ISAAC_SIM_ROOT or ISAAC_SIM_PYTHON for this host." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/install/setup.bash" ]]; then
  echo "[FAIL] ROS workspace is not built: ${PROJECT_ROOT}/install/setup.bash" >&2
  exit 1
fi
if [[ ! -x "${SOCIAL_ROOT}/scripts/run_koch_stream.sh" ]]; then
  echo "[FAIL] social-safety-decision runtime not found: ${SOCIAL_ROOT}" >&2
  exit 1
fi
if [[ -z "${DISPLAY:-}" && "${OPENARM_ALLOW_NO_DISPLAY:-0}" != "1" ]]; then
  echo "[FAIL] DISPLAY is unset; visible Isaac Sim requires a desktop session." >&2
  exit 1
fi

"${PROJECT_ROOT}/scripts/stop_openarm_demo.sh" --quiet || true

set +u
source /opt/ros/humble/setup.bash
source "${PROJECT_ROOT}/install/setup.bash"
set -u

# Humble installations assembled from the minimal MoveIt packages may omit
# this runtime plugin. Cache the official Debian payload locally without
# requiring root so the one-click launch remains reproducible on this host.
if ! ros2 pkg prefix moveit_simple_controller_manager >/dev/null 2>&1; then
  MOVEIT_RUNTIME="${STATE_DIR}/moveit_runtime/opt/ros/humble"
  if [[ ! -f "${MOVEIT_RUNTIME}/share/moveit_simple_controller_manager/local_setup.bash" ]]; then
    mkdir -p "${STATE_DIR}/moveit_runtime"
    MOVEIT_DEB="$(find /tmp -maxdepth 2 -type f -name 'ros-humble-moveit-simple-controller-manager_*.deb' -print -quit 2>/dev/null || true)"
    if [[ -z "${MOVEIT_DEB}" ]]; then
      (
        cd "${STATE_DIR}"
        apt-get download ros-humble-moveit-simple-controller-manager >/dev/null
      )
      MOVEIT_DEB="$(find "${STATE_DIR}" -maxdepth 1 -type f -name 'ros-humble-moveit-simple-controller-manager_*.deb' -print -quit)"
    fi
    dpkg-deb -x "${MOVEIT_DEB}" "${STATE_DIR}/moveit_runtime"
  fi
  export AMENT_CURRENT_PREFIX="${MOVEIT_RUNTIME}"
  set +u
  source "${MOVEIT_RUNTIME}/share/moveit_simple_controller_manager/local_setup.bash"
  set -u
  unset AMENT_CURRENT_PREFIX
fi

export OPENARM_SIM_ROOT="${PROJECT_ROOT}"
export ISAAC_SIM_PYTHON
export CAMERA_INPUT_TOPIC="${CAMERA_INPUT_TOPIC:-/rgbd/color/image_raw}"
export CAMERA_QOS="${CAMERA_QOS:-sensor_data}"
export KOCH_CONFIGURE_CAMERA=0
export KOCH_ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"

if [[ -f "${SOCIAL_ROOT}/scripts/ros2_lan_env.sh" ]]; then
  set +u
  source "${SOCIAL_ROOT}/scripts/ros2_lan_env.sh"
  set -u
fi

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

systemctl --user set-environment \
  OPENARM_SIM_ROOT="${PROJECT_ROOT}" \
  CAMERA_INPUT_TOPIC="${CAMERA_INPUT_TOPIC}" \
  CAMERA_QOS="${CAMERA_QOS}" \
  KOCH_CONFIGURE_CAMERA="${KOCH_CONFIGURE_CAMERA}" \
  KOCH_ROS_DOMAIN_ID="${KOCH_ROS_DOMAIN_ID}"
systemctl --user restart realtime-safety-3d.service

RUN_ID="demo_$(date +%Y%m%d_%H%M%S)"
export OPENARM_RUN_ID="${RUN_ID}"
setsid ros2 launch openarm_sim_bringup ground_truth_validation.launch.py \
  scenario:=no_obstacle \
  headless:=false \
  use_rviz:=false \
  auto_start:=false \
  output_root:="${PROJECT_ROOT}/results" \
  >"${LOG_DIR}/openarm_demo.log" 2>&1 &
LAUNCH_PID=$!
echo "${LAUNCH_PID}" >"${LAUNCH_PID_FILE}"

if [[ "${OPENARM_DEMO_OPEN_BROWSER:-1}" == "1" ]]; then
  (
    for _ in $(seq 1 120); do
      if curl -fsS --max-time 1 "${GUI_URL}" >/dev/null 2>&1; then
        xdg-open "${GUI_URL}" >/dev/null 2>&1 || true
        break
      fi
      sleep 1
    done
  ) &
fi

topic_has_message() {
  timeout 4 ros2 topic echo --once "$1" >/dev/null 2>&1
}

controller_ready() {
  ros2 action info /left_joint_trajectory_controller/follow_joint_trajectory 2>/dev/null |
    grep -q 'Action servers: 1'
}

dynamic_avoidance_ready() {
  timeout 4 ros2 topic echo --once /openarm/dynamic_avoidance/status 2>/dev/null |
    grep -q 'READY'
}

deadline=$((SECONDS + READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "[FAIL] OpenArm launch exited; inspect ${LOG_DIR}/openarm_demo.log" >&2
    exit 1
  fi
  gui_http=false
  curl -fsS --max-time 2 "${GUI_URL}" >/dev/null 2>&1 && gui_http=true
  if pgrep -f 'scripts/run_sim.py.*--no-headless' >/dev/null \
    && topic_has_message /clock \
    && topic_has_message /joint_states \
    && topic_has_message /rgbd/color/image_raw \
    && topic_has_message /rgbd/aligned_depth_to_color/image_raw \
    && topic_has_message /realtime_safety/camera/image_raw \
    && [[ "${gui_http}" == true ]] \
    && controller_ready \
    && dynamic_avoidance_ready; then
    echo "[OK] Isaac Sim GUI"
    echo "[OK] /clock"
    echo "[OK] /joint_states"
    echo "[OK] RGB frame received"
    echo "[OK] Depth frame received"
    echo "[OK] Social Safety received RGB"
    echo "[OK] 8080 GUI rendered live image"
    echo "[OK] Controller"
    echo "[OK] Dynamic Avoidance"
    echo "OPENARM DEMO READY"
    echo "GUI: ${GUI_URL}"
    exit 0
  fi
  sleep 2
done

echo "[FAIL] Demo readiness timed out after ${READY_TIMEOUT}s." >&2
echo "[INFO] Log: ${LOG_DIR}/openarm_demo.log" >&2
exit 1
