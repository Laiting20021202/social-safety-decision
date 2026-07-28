#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/scripts/ros2_lan_env.sh"
set -u

exec rviz2 \
  -d "${ROOT_DIR}/configs/realtime_safety.rviz" \
  -f realtime_safety_frame
