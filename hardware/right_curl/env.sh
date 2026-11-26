#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
if [[ -f /opt/laiting-camera/local_setup.bash ]]; then source /opt/laiting-camera/local_setup.bash; fi
export AMENT_PREFIX_PATH="/opt/right-curl:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/opt/right-curl/lib:${LD_LIBRARY_PATH:-}"
exec "$@"
