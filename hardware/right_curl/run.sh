#!/usr/bin/env bash
# Laboratory entry points. Motion is exclusively streamed by the guarded client.
set -euo pipefail
guarded=$(cd -- "$(dirname -- "$0")/../guarded_curl" && pwd)
role=${1:-}
shift || true
case "$role" in
  robot) exec bash "$guarded/gateway.sh" "$@" ;;
  workstation) exec bash "$guarded/station.sh" up ;;
  ros2) exec bash "$guarded/station.sh" ros2 "$@" ;;
  status) exec bash "$guarded/station.sh" run observe --chest-only "$@" ;;
  curl) exec bash "$guarded/station.sh" run repeat --execute --cycles 1 "$@" ;;
  *) echo "Usage: $0 {robot|workstation|status|ros2|curl}" >&2; exit 2 ;;
esac
