#!/usr/bin/env bash
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/isaac_ros_env.sh"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" app.py \
  --profile isaac_lan \
  --source ros2:///camera/front/color/image_raw/compressed \
  --camera-qos sensor_data --ros-domain-id 0 \
  --pointcloud-topic /realtime_safety/pointcloud \
  --pointcloud-frame-id isaac_front_estimated_optical_frame \
  --pointcloud-coordinate-mode ros_optical --pointcloud-rate 8 \
  --no-log "$@"
