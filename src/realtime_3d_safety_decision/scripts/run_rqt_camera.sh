#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u
# shellcheck source=ros2_lan_env.sh
source "${ROOT_DIR}/scripts/ros2_lan_env.sh"

TOPIC="${1:-/realtime_safety/camera/image_raw}"
exec ros2 run rqt_image_view rqt_image_view "${TOPIC}"
