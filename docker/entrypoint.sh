#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
if [[ -f /opt/social-safety-ros/setup.bash ]]; then
  source /opt/social-safety-ros/setup.bash
fi

if [[ "$1" == /isaac-sim/python.sh && "${ACCEPT_EULA:-}" != Y ]]; then
  echo "Set ACCEPT_EULA=Y after reviewing the NVIDIA Omniverse license." >&2
  exit 2
fi

exec "$@"
