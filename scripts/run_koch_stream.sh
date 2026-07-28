#!/usr/bin/env bash
set -euo pipefail

ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
if [[ -f "${ROS_SETUP_FILE}" ]]; then
  set +u
  source "${ROS_SETUP_FILE}"
  set -u
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

CAMERA_SOURCE="${KOCH_CAMERA_SOURCE:-${KOCH_STREAM_URL:-http://192.168.0.231:8080/stream?topic=/koch_remote/camera/image_raw&type=mjpeg&width=320&height=240&quality=50&qos_profile=sensor_data}}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/realtime_safety/pointcloud}"
POINTCLOUD_FRAME="${POINTCLOUD_FRAME:-realtime_safety_frame}"
POINTCLOUD_RATE="${POINTCLOUD_RATE:-12}"
YOLO_OBSTACLE_POINTCLOUD_TOPIC="${YOLO_OBSTACLE_POINTCLOUD_TOPIC:-/realtime_safety/yolo_obstacles/pointcloud}"
YOLO_OBSTACLE_POINTCLOUD_RATE="${YOLO_OBSTACLE_POINTCLOUD_RATE:-12}"
ARM_OBSTACLE_RELATIONSHIP_TOPIC="${ARM_OBSTACLE_RELATIONSHIP_TOPIC:-/realtime_safety/arm_obstacle_relationships}"
ARM_OBSTACLE_RELATIONSHIP_RATE="${ARM_OBSTACLE_RELATIONSHIP_RATE:-12}"
CAMERA_PREVIEW_TOPIC="${CAMERA_PREVIEW_TOPIC:-/realtime_safety/camera/image_raw}"
CAMERA_PREVIEW_RATE="${CAMERA_PREVIEW_RATE:-10}"
ROS_DOMAIN_ID="${KOCH_ROS_DOMAIN_ID:-42}"
ROS_LAN_PEER="${ROS_LAN_PEER:-192.168.0.231}"
export ROS_DOMAIN_ID ROS_LAN_PEER
# shellcheck source=ros2_lan_env.sh
source "${ROOT_DIR}/scripts/ros2_lan_env.sh"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  echo "Cannot determine the LAN interface/IP used to reach ${ROS_LAN_PEER}." >&2
  exit 1
fi

cd "${ROOT_DIR}"
if [[ "${KOCH_CONFIGURE_CAMERA:-1}" != "0" ]]; then
  "${PYTHON}" scripts/configure_koch_camera.py \
    --output-encoding "${KOCH_CAMERA_ENCODING:-yuv422_yuy2}" \
    --timeout "${KOCH_CAMERA_CONFIG_TIMEOUT:-12}" ||
    echo "Camera parameter setup was not available; continuing with the camera's current encoding." >&2
fi
exec "${PYTHON}" app.py \
  --source "${CAMERA_SOURCE}" \
  --profile koch_lan \
  --device cuda \
  --host 0.0.0.0 \
  --pointcloud-topic "${POINTCLOUD_TOPIC}" \
  --pointcloud-frame-id "${POINTCLOUD_FRAME}" \
  --pointcloud-rate "${POINTCLOUD_RATE}" \
  --pointcloud-coordinate-mode camera_y_forward \
  --yolo-obstacle-pointcloud-topic "${YOLO_OBSTACLE_POINTCLOUD_TOPIC}" \
  --yolo-obstacle-pointcloud-rate "${YOLO_OBSTACLE_POINTCLOUD_RATE}" \
  --arm-obstacle-relationship-topic "${ARM_OBSTACLE_RELATIONSHIP_TOPIC}" \
  --arm-obstacle-relationship-rate "${ARM_OBSTACLE_RELATIONSHIP_RATE}" \
  --camera-preview-topic "${CAMERA_PREVIEW_TOPIC}" \
  --camera-preview-rate "${CAMERA_PREVIEW_RATE}" \
  --ros-domain-id "${ROS_DOMAIN_ID}" \
  "$@"
