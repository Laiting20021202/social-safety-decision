#!/usr/bin/env bash
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/isaac_ros_env.sh"
cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" -m realtime_safety.ros2_bridge.tidybot_retreat "$@"
