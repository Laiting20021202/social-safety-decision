#!/usr/bin/env bash
set -euo pipefail

ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
if [[ -f "${ROS_SETUP_FILE}" ]]; then
  set +u
  source "${ROS_SETUP_FILE}"
  set -u
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROS_SETUP="${ROOT_DIR}/install/setup.bash"
if [[ -f "${LOCAL_ROS_SETUP}" ]]; then
  set +u
  source "${LOCAL_ROS_SETUP}"
  set -u
fi
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

# NumPy/OpenCV/PyTorch otherwise create one worker per CPU core.  At the
# camera's 320x240 resolution that fan-out costs more than the operation and
# makes frame delivery bursty.  Export the limits before either Python process
# imports its numerical libraries so their native thread pools inherit them.
export OPENBLAS_NUM_THREADS="${REALTIME_OPENBLAS_THREADS:-2}"
export OMP_NUM_THREADS="${REALTIME_OMP_THREADS:-4}"
export MKL_NUM_THREADS="${REALTIME_MKL_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${REALTIME_NUMEXPR_THREADS:-4}"

# Camera input is configurable and defaults to the generic simulator RGB-D
# topic.  Existing Koch deployments can retain their source by exporting
# KOCH_CAMERA_SOURCE or CAMERA_SOURCE before launching this script.
CAMERA_INPUT_TOPIC="${CAMERA_INPUT_TOPIC:-${KOCH_CAMERA_TOPIC:-/rgbd/color/image_raw}}"
CAMERA_SOURCE="${CAMERA_SOURCE:-${KOCH_CAMERA_SOURCE:-ros2://${CAMERA_INPUT_TOPIC}}}"
CAMERA_QOS="${CAMERA_QOS:-sensor_data}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/realtime_safety/pointcloud}"
POINTCLOUD_FRAME="${POINTCLOUD_FRAME:-realtime_safety_frame}"
POINTCLOUD_RATE="${POINTCLOUD_RATE:-15}"
POINTCLOUD_COORDINATE_MODE="${POINTCLOUD_COORDINATE_MODE:-camera_y_forward}"
YOLO_OBSTACLE_CANDIDATE_TOPIC="${YOLO_OBSTACLE_CANDIDATE_TOPIC:-/realtime_safety/yolo_obstacles/candidate_cloud}"
YOLO_OBSTACLE_POINTCLOUD_RATE="${YOLO_OBSTACLE_POINTCLOUD_RATE:-12}"
ARM_OBSTACLE_RELATIONSHIP_TOPIC="${ARM_OBSTACLE_RELATIONSHIP_TOPIC:-/realtime_safety/arm_obstacle_relationships}"
ARM_OBSTACLE_RELATIONSHIP_RATE="${ARM_OBSTACLE_RELATIONSHIP_RATE:-12}"
CAMERA_PREVIEW_TOPIC="${CAMERA_PREVIEW_TOPIC:-/realtime_safety/camera/image_raw}"
CAMERA_PREVIEW_RATE="${CAMERA_PREVIEW_RATE:-12}"
CAMERA_PREVIEW_FRAME="${CAMERA_PREVIEW_FRAME:-realtime_safety_frame}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/realtime_safety/camera/camera_info}"
EDGETAM_TRACKER_ENABLE="${EDGETAM_TRACKER_ENABLE:-1}"
EDGETAM_CONFIG="${EDGETAM_CONFIG:-${ROOT_DIR}/config/edgetam_pointcloud_tracker.yaml}"
ROS_DOMAIN_ID="${KOCH_ROS_DOMAIN_ID:-42}"
ROS_LAN_PEER="${ROS_LAN_PEER:-192.168.0.231}"
export ROS_DOMAIN_ID ROS_LAN_PEER
if [[ "${CAMERA_INPUT_TOPIC}" == /rgbd/* && "${POINTCLOUD_FRAME}" == "realtime_safety_frame" ]]; then
  POINTCLOUD_FRAME=rgbd_color_optical_frame
fi
if [[ "${CAMERA_INPUT_TOPIC}" == /rgbd/* && "${POINTCLOUD_COORDINATE_MODE}" == "camera_y_forward" ]]; then
  POINTCLOUD_COORDINATE_MODE=ros_optical
fi
# shellcheck source=ros2_lan_env.sh
source "${ROOT_DIR}/scripts/ros2_lan_env.sh"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  echo "Cannot determine the LAN interface/IP used to reach ${ROS_LAN_PEER}." >&2
  exit 1
fi

cd "${ROOT_DIR}"
if [[ "${EDGETAM_TRACKER_ENABLE}" != "0" && ! -f "${LOCAL_ROS_SETUP}" ]]; then
  echo "EdgeTAM ROS interfaces are not built. Run: colcon build --symlink-install" >&2
  exit 1
fi
CONFIGURE_CAMERA_DEFAULT=1
[[ "${CAMERA_INPUT_TOPIC}" == /rgbd/* ]] && CONFIGURE_CAMERA_DEFAULT=0
if [[ "${KOCH_CONFIGURE_CAMERA:-${CONFIGURE_CAMERA_DEFAULT}}" != "0" ]]; then
  "${PYTHON}" scripts/configure_koch_camera.py \
    --output-encoding "${KOCH_CAMERA_ENCODING:-yuv422_yuy2}" \
    --timeout "${KOCH_CAMERA_CONFIG_TIMEOUT:-12}" ||
    echo "Camera parameter setup was not available; continuing with the camera's current encoding." >&2
fi

app_args=(
  --source "${CAMERA_SOURCE}" \
  --camera-qos "${CAMERA_QOS}" \
  --profile koch_lan \
  --device cuda \
  --host 0.0.0.0 \
  --pointcloud-topic "${POINTCLOUD_TOPIC}" \
  --pointcloud-frame-id "${POINTCLOUD_FRAME}" \
  --pointcloud-rate "${POINTCLOUD_RATE}" \
  --pointcloud-coordinate-mode "${POINTCLOUD_COORDINATE_MODE}" \
  --yolo-obstacle-pointcloud-topic "${YOLO_OBSTACLE_CANDIDATE_TOPIC}" \
  --yolo-obstacle-pointcloud-rate "${YOLO_OBSTACLE_POINTCLOUD_RATE}" \
  --arm-obstacle-relationship-topic "${ARM_OBSTACLE_RELATIONSHIP_TOPIC}" \
  --arm-obstacle-relationship-rate "${ARM_OBSTACLE_RELATIONSHIP_RATE}" \
  --camera-preview-topic "${CAMERA_PREVIEW_TOPIC}" \
  --camera-preview-rate "${CAMERA_PREVIEW_RATE}" \
  --camera-preview-frame-id "${CAMERA_PREVIEW_FRAME}" \
  --camera-info-topic "${CAMERA_INFO_TOPIC}" \
  --ros-domain-id "${ROS_DOMAIN_ID}" \
)
app_args+=("$@")

edge_pid=""
app_pid=""
terminate_children() {
  trap - INT TERM
  if [[ -n "${app_pid}" ]] && kill -0 "${app_pid}" 2>/dev/null; then
    kill "${app_pid}" 2>/dev/null || true
  fi
  if [[ -n "${edge_pid}" ]] && kill -0 "${edge_pid}" 2>/dev/null; then
    kill "${edge_pid}" 2>/dev/null || true
  fi
  [[ -z "${app_pid}" ]] || wait "${app_pid}" 2>/dev/null || true
  [[ -z "${edge_pid}" ]] || wait "${edge_pid}" 2>/dev/null || true
}
trap terminate_children EXIT INT TERM

if [[ "${EDGETAM_TRACKER_ENABLE}" != "0" ]]; then
  edge_args=(
    --ros-args
    --params-file "${EDGETAM_CONFIG}"
    -p compatibility.publish_legacy_obstacle_alias:=false
  )
if [[ "${CAMERA_INPUT_TOPIC}" == /rgbd/* ]]; then
    app_args+=(
      --depth-mode rgbd
      --scale-mode rgbd
      --rgbd-color-topic /rgbd/color/image_raw
      --rgbd-depth-topic /rgbd/aligned_depth_to_color/image_raw
      --rgbd-camera-info-input-topic /rgbd/color/camera_info
      --rgbd-camera-pose-topic /sim/camera/pose
      --rgbd-generated-world-pointcloud-topic /realtime_safety/environment_cloud_world
      --sim-config-root "${OPENARM_SIM_ROOT:?OPENARM_SIM_ROOT is required for simulator RGB-D mode}/config"
    )
    # Generate geometry from synchronized RGB + aligned depth inside the
    # 3D-safety process. Gazebo's native PointCloud2 is deliberately not an
    # input to perception.
    edge_args+=(
      -p topics.rgb_image:=/rgbd/color/image_raw
      -p topics.depth_image:=/rgbd/aligned_depth_to_color/image_raw
      -p topics.camera_info:=/rgbd/color/camera_info
      -p "topics.pointcloud:=''"
      -p use_sim_time:=true
      -p frames.tracking_frame:=rgbd_color_optical_frame
      -p frames.robot_base_frame:=world
      -p frames.camera_frame:=rgbd_color_optical_frame
      -p workspace.min_x:=-1.5
      -p workspace.max_x:=1.5
      -p workspace.min_y:=-1.2
      -p workspace.max_y:=1.2
      -p workspace.min_z:=0.1
      -p workspace.max_z:=3.0
      -p background.depth_axis:=2
      -p background.horizontal_axis:=0
      -p background.vertical_axis:=1
      -p projection.camera_axis_mode:=ros_optical
      -p sync.sensor_stale_timeout_sec:=3.0
      # Remove OpenArm returns before foreground clustering so the moving
      # white/black grippers cannot become hand candidates.
      -p self_filter.enabled:=true
      -p self_filter.link_frames:="[openarm_left_link0,openarm_left_link1,openarm_left_link2,openarm_left_link3,openarm_left_link4,openarm_left_link5,openarm_left_link6,openarm_left_link7,openarm_left_hand,openarm_left_left_finger,openarm_left_right_finger,openarm_right_link0,openarm_right_link1,openarm_right_link2,openarm_right_link3,openarm_right_link4,openarm_right_link5,openarm_right_link6,openarm_right_link7,openarm_right_hand,openarm_right_left_finger,openarm_right_right_finger]"
      -p self_filter.link_radii_m:="[0.080,0.115,0.094,0.186,0.119,0.136,0.076,0.099,0.085,0.093,0.093,0.080,0.115,0.094,0.186,0.119,0.136,0.076,0.099,0.085,0.093,0.093]"
      -p self_filter.padding:=0.005
      -p self_filter.fail_closed:=true
      -p hand_candidate.enabled:=true
      # The Gazebo demo uses the textured LibHand mesh.  Keep the real RGB
      # MediaPipe semantic gate enabled; geometry alone is never promoted to a
      # perception obstacle.
      -p hand_semantics.enabled:=true
    )
  fi
  "${PYTHON}" scripts/edgetam_pointcloud_tracker_node "${edge_args[@]}" &
  edge_pid=$!
fi

"${PYTHON}" app.py "${app_args[@]}" &
app_pid=$!

set +e
if [[ -n "${edge_pid}" ]]; then
  wait -n "${app_pid}" "${edge_pid}"
else
  wait "${app_pid}"
fi
status=$?
set -e
terminate_children
trap - EXIT INT TERM
exit "${status}"
