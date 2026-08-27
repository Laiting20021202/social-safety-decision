#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SOCIAL_ROOT="${SOCIAL_SAFETY_ROOT:-${REPOSITORY_ROOT}}"
STATE_DIR="${OPENARM_DEMO_STATE_DIR:-${PROJECT_ROOT}/.demo}"
LOG_DIR="${STATE_DIR}/logs"
LAUNCH_PID_FILE="${STATE_DIR}/launch.pid"
READY_TIMEOUT="${OPENARM_DEMO_READY_TIMEOUT:-120}"
GUI_URL="${OPENARM_GUI_URL:-http://192.168.0.234:8080/}"
HAND_PREVIEW="${OPENARM_DEMO_HAND_PREVIEW:-1}"
mkdir -p "${LOG_DIR}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[FAIL] DISPLAY is unset; visible Gazebo + RViz requires a desktop session." >&2
  exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[FAIL] ROS 2 Humble is not installed." >&2
  exit 1
fi
if [[ ! -x "${SOCIAL_ROOT}/scripts/run_koch_stream.sh" ]]; then
  echo "[FAIL] social-safety-decision runtime not found: ${SOCIAL_ROOT}" >&2
  exit 1
fi

"${PROJECT_ROOT}/scripts/stop_openarm_demo.sh" --quiet || true
export OPENARM_SIM_ROOT="${PROJECT_ROOT}"
python3 "${PROJECT_ROOT}/scripts/create_gazebo_world.py" >/dev/null
python3 "${PROJECT_ROOT}/scripts/prepare_gazebo_robot.py" >/dev/null

set +u
source /opt/ros/humble/setup.bash
set -u

# The host's minimal MoveIt installation omits the simple controller-manager
# plugin. Reuse the locally extracted official Humble package so MoveIt can
# hand planned trajectories to the Gazebo FollowJointTrajectory controllers.
MOVEIT_RUNTIME="${STATE_DIR}/moveit_runtime/opt/ros/humble"
if ! ros2 pkg prefix moveit_simple_controller_manager >/dev/null 2>&1; then
  if [[ ! -f "${MOVEIT_RUNTIME}/share/moveit_simple_controller_manager/local_setup.bash" ]]; then
    echo "[FAIL] missing local MoveIt controller runtime: ${MOVEIT_RUNTIME}" >&2
    echo "[INFO] Run scripts/start_openarm_demo.sh once to prepare it." >&2
    exit 1
  fi
  export AMENT_CURRENT_PREFIX="${MOVEIT_RUNTIME}"
  set +u
  source "${MOVEIT_RUNTIME}/share/moveit_simple_controller_manager/local_setup.bash"
  set -u
  unset AMENT_CURRENT_PREFIX
fi

colcon --log-base "${PROJECT_ROOT}/log" build --symlink-install \
  --packages-select \
    openarm_dynamic_avoidance \
    openarm_perception_adapter \
    openarm_safety_bridge \
    openarm_sim_bringup \
    openarm_sorting_task \
  --base-paths "${PROJECT_ROOT}/ros2_ws/src" \
  --build-base "${PROJECT_ROOT}/build" \
  --install-base "${PROJECT_ROOT}/install" \
  >"${LOG_DIR}/gazebo_build.log" 2>&1
set +u
source "${PROJECT_ROOT}/install/setup.bash"
source "${SOCIAL_ROOT}/scripts/ros2_lan_env.sh"
set -u

export CAMERA_INPUT_TOPIC=/rgbd/color/image_raw
export CAMERA_QOS=sensor_data
export KOCH_CONFIGURE_CAMERA=0
export KOCH_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
systemctl --user set-environment \
  OPENARM_SIM_ROOT="${PROJECT_ROOT}" \
  CAMERA_INPUT_TOPIC="${CAMERA_INPUT_TOPIC}" \
  CAMERA_QOS="${CAMERA_QOS}" \
  KOCH_CONFIGURE_CAMERA="${KOCH_CONFIGURE_CAMERA}" \
  KOCH_ROS_DOMAIN_ID="${KOCH_ROS_DOMAIN_ID}"
systemctl --user restart realtime-safety-3d.service

setsid ros2 launch openarm_sim_bringup gazebo_phase1.launch.py gui:=true rviz:=true \
  >"${LOG_DIR}/gazebo_demo.log" 2>&1 &
LAUNCH_PID=$!
echo "${LAUNCH_PID}" >"${LAUNCH_PID_FILE}"

if [[ "${OPENARM_DEMO_OPEN_BROWSER:-1}" == "1" ]]; then
  (
    for _ in $(seq 1 60); do
      if curl -fsS --max-time 1 "${GUI_URL}" >/dev/null 2>&1; then
        xdg-open "${GUI_URL}" >/dev/null 2>&1 || true
        break
      fi
      sleep 1
    done
  ) &
fi

deadline=$((SECONDS + READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "[FAIL] Gazebo launch exited; inspect ${LOG_DIR}/gazebo_demo.log" >&2
    exit 1
  fi
  if pgrep -x gzclient >/dev/null \
    && pgrep -x rviz2 >/dev/null \
    && curl -fsS --max-time 2 "${GUI_URL}" >/dev/null 2>&1; then
    if python3 "${PROJECT_ROOT}/scripts/validate_gazebo_rgbd.py" \
      --seconds 3 \
      --require-gui-preview \
      --output "${STATE_DIR}/health" \
      >"${LOG_DIR}/gazebo_health.log" 2>&1; then
      if ! python3 "${PROJECT_ROOT}/scripts/validate_rgbd_realtime.py" \
        --duration 3 \
        >"${LOG_DIR}/rgbd_reprojection_health.log" 2>&1; then
        sleep 2
        continue
      fi
      if ! python3 "${PROJECT_ROOT}/scripts/validate_gazebo_control_stack.py" \
        --seconds 25 \
        --output "${STATE_DIR}/health/control_stack.json" \
        >"${LOG_DIR}/gazebo_control_health.log" 2>&1; then
        sleep 2
        continue
      fi
      if ! python3 "${PROJECT_ROOT}/scripts/validate_rgbd_occlusion.py" \
        >"${LOG_DIR}/rgbd_occlusion_health.log" 2>&1; then
        sleep 2
        continue
      fi
      if [[ "${HAND_PREVIEW}" == "1" ]]; then
        ros2 topic pub --once /sim/hand/command std_msgs/msg/String \
          "{data: 'scenario:perception_preview'}" \
          >"${LOG_DIR}/hand_preview_command.log" 2>&1
        ros2 topic pub --once /sim/hand/command std_msgs/msg/String \
          "{data: 'trigger'}" \
          >>"${LOG_DIR}/hand_preview_command.log" 2>&1
        if ! python3 "${PROJECT_ROOT}/scripts/validate_rgbd_realtime.py" \
          --duration 12 \
          --require-obstacle \
          >"${LOG_DIR}/hand_obstacle_health.log" 2>&1; then
          sleep 2
          continue
        fi
        # The startup hand exists only to prove the real RGB -> model ->
        # depth-cloud path.  Park it before READY so the safety supervisor
        # does not leave every GUI motion command blocked in PAUSE.
        ros2 topic pub --once /sim/hand/command std_msgs/msg/String \
          "{data: 'withdraw'}" \
          >>"${LOG_DIR}/hand_preview_command.log" 2>&1
        sleep 4
        ros2 topic pub --once /openarm/safety/command std_msgs/msg/String \
          "{data: 'reset'}" \
          >>"${LOG_DIR}/hand_preview_command.log" 2>&1
        ros2 topic pub --once /openarm/task/command std_msgs/msg/String \
          "{data: 'reset'}" \
          >>"${LOG_DIR}/hand_preview_command.log" 2>&1
        sleep 2
        if ! python3 "${PROJECT_ROOT}/scripts/validate_gazebo_control_stack.py" \
          --seconds 10 \
          --output "${STATE_DIR}/health/control_after_preview.json" \
          >"${LOG_DIR}/gazebo_control_after_preview.log" 2>&1; then
          sleep 2
          continue
        fi
      fi
      echo "[OK] Gazebo GUI"
      echo "[OK] RViz2"
      echo "[OK] /clock + /joint_states"
      echo "[OK] RGB + 32FC1 Depth"
      echo "[OK] 3D Safety image-generated optical + cropped world point clouds"
      echo "[OK] RGB/depth/generated-cloud stamps + metric back-projection"
      echo "[OK] Current-frame depth occlusion (no hidden/stale cube points)"
      echo "[OK] MediaPipe + EdgeTAM models ready"
      echo "[OK] Social Safety received RGB + published obstacle cloud"
      echo "[OK] 8080 GUI rendered live image"
      echo "[OK] MoveIt + OpenArm controller"
      echo "[OK] Dynamic Avoidance (perception source)"
      if [[ "${HAND_PREVIEW}" == "1" ]]; then
        echo "[OK] Real-model hand obstacle point cloud"
      fi
      echo "OPENARM GAZEBO DEMO READY"
      echo "GUI: ${GUI_URL}"
      exit 0
    fi
  fi
  sleep 2
done

echo "[FAIL] Gazebo Phase-1 readiness timed out after ${READY_TIMEOUT}s." >&2
echo "[INFO] ${LOG_DIR}/gazebo_demo.log" >&2
echo "[INFO] ${LOG_DIR}/gazebo_health.log" >&2
exit 1
