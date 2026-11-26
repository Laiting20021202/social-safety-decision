#!/usr/bin/env bash
# Run on the NUC. Rebind only the dedicated D415 after USB re-enumeration.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
kind=${1:-d415}
case "$kind" in
  d415) serial=411543020124 ;;
  d435) serial=349143021052 ;;
  *) echo "Expected d415 or d435" >&2; exit 2 ;;
esac
camera=laiting-openarm-$kind
gateway=laiting-openarm-gateway
label=org.laiting.$kind-devices
devices=()
for usb in /sys/bus/usb/devices/*; do
  [[ -r "$usb/serial" && "$(<"$usb/serial")" == "$serial" ]] || continue
  path=$(realpath "$usb")
  devices+=("$(printf '/dev/bus/usb/%03d/%03d' "$(<"$usb/busnum")" "$(<"$usb/devnum")")")
  for node in /sys/class/video4linux/* "$path"/*/media[0-9]*; do
    [[ -e "$node" && "$(realpath "$node")" == "$path/"* ]] || continue
    devices+=("/dev/${node##*/}")
  done
  break
done
(( ${#devices[@]} >= 4 )) || { echo "$kind unavailable on NUC" >&2; exit 1; }
spec="${devices[*]}"
if docker container inspect "$camera" >/dev/null 2>&1; then
  orig=$(docker inspect --format "{{ index .Config.Labels \"$label\" }}" "$camera")
  [[ -n "$orig" ]] || { echo "Unexpected container owner: $camera" >&2; exit 1; }
  if [[ "$orig" == "$spec" ]]; then docker start "$camera"; exit 0; fi
  docker rm -f "$camera" >/dev/null
fi
# Retire only this project's old embedded camera process. The encoder and CAN
# processes remain alive, including any controller supporting an elevated arm.
if [[ "$(docker inspect -f '{{.State.Running}}' "$gateway" 2>/dev/null || true)" == true ]]; then
  docker exec -i "$gateway" python3 - "$kind" <<'PY'
import os, signal, sys, time
from pathlib import Path
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv = (proc/'cmdline').read_bytes().split(b'\0')
        embedded_rgbd = argv[0].endswith(b'/realsense2_camera_node') and b'__node:=d415' in argv
        embedded_relay = b'/opt/right-curl/app/camera_relay.py' in argv
        if embedded_rgbd and sys.argv[1]=='d415' or embedded_relay and sys.argv[1]=='d435':
            os.kill(int(proc.name), signal.SIGINT)
            for _ in range(60):
                if not proc.exists(): break
                time.sleep(0.1)
            else: raise SystemExit("Embedded camera process still running")
    except (FileNotFoundError, ProcessLookupError):
        pass
PY
fi
args=()
for device in "${devices[@]}"; do args+=(--device "$device"); done
if [[ "$kind" == d415 ]]; then
  driver=(ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/camera -r __node:=d415 --params-file /opt/laiting-camera-config/d415.yaml)
else
  driver=(python3 /opt/laiting-camera-config/d435_source.py)
fi
docker run -d --name "$camera" --init --network host --cap-drop ALL --security-opt no-new-privileges \
  --label "$label=$spec" -e "ROS_DOMAIN_ID=${OPENARM_ROS_DOMAIN_ID:-88}" "${args[@]}" \
  --log-opt max-size=8m --log-opt max-file=2 -v /run/udev:/run/udev:ro \
  -v "$here:/opt/laiting-camera-config:ro" laiting-openarm-robot:humble "${driver[@]}"
