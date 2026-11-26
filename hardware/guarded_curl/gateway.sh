#!/usr/bin/env bash
# Fixed-base gateway and independently rebindable chest D415.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
name=laiting-openarm-gateway
mount_pose=()
pose_arg=()
if [[ -n ${OPENARM_TABLETOP_PROFILE:-} ]]; then
  source_pose=$(realpath -e "$OPENARM_TABLETOP_PROFILE")
  digest=$(sha256sum "$source_pose" | cut -d' ' -f1)
  mount_pose=(--label "org.laiting.tabletop=$digest" -v "$source_pose:/opt/laiting-tabletop.json:ro")
  pose_arg=(tabletop_profile:=/opt/laiting-tabletop.json)
  if docker container inspect "$name" >/dev/null 2>&1 &&
     [[ $(docker inspect -f '{{index .Config.Labels "org.laiting.tabletop"}}' "$name") != "$digest" ]]; then
    echo "Gateway has a different pose. After onsite handoff and supported shutdown, recreate the project gateway with the surveyed profile." >&2
    exit 1
  fi
fi
if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
  if docker container inspect "$name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.Config.Image}}' "$name")" != "laiting-openarm-robot:humble" ]]; then
      echo "Unexpected image for $name" >&2; exit 1
    fi
    if [[ "$(docker inspect -f '{{index .Config.Labels \"org.laiting.role\"}}' "$name")" == 'encoders' ]]; then
      docker start "$name" >/dev/null
    else
      # The stopped legacy gateway carries device numbers that may have disappeared.
      docker rename "$name" "$name-before-split-$(date +%s)"
    fi
  fi
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    docker run -d --init --network host --name "$name" --cap-drop ALL --security-opt no-new-privileges \
      -e "ROS_DOMAIN_ID=${OPENARM_ROS_DOMAIN_ID:-88}" --label org.laiting.role=encoders \
      --log-opt max-size=10m --log-opt max-file=2 \
      -v "$here/../right_curl/right.launch.py:/opt/right-curl/app/right.launch.py:ro" \
      "${mount_pose[@]}" laiting-openarm-robot:humble ros2 launch /opt/right-curl/app/right.launch.py embedded_cameras:=false "${pose_arg[@]}"
  fi
fi
bash "$here/camera.sh" d415
