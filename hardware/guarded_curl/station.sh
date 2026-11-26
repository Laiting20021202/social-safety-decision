#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=laiting-openarm-station
image=${STATION_IMAGE:-laiting-openarm-safety:humble}
case "${1:-up}" in
  build)
    ctx=$(mktemp -d)
    trap 'rm -rf -- "$ctx"' EXIT
    mkdir -p "$ctx"/{app,semantic,moveit}
    cp "$root"/src/realtime_3d_safety_decision/realtime_safety/edgetam_tracker/{models,hand_semantic_gate}.py "$ctx/semantic/"
    moveit="$root/src/openarm_bimanual_moveit_config/config/openarm_v1.0"
    if [[ -r "$moveit/openarm_bimanual.srdf" ]]; then
      cp "$moveit"/{openarm_bimanual.srdf,kinematics.yaml} "$ctx/moveit/"
    else
      docker run --rm --network none --user "$(id -u):$(id -g)" -v "$ctx/moveit:/stage" \
        --entrypoint bash social-safety-decision-control:humble -ec \
        'cp /opt/social-safety-decision/.openarm-vendor/openarm_ros2/openarm_bimanual_moveit_config/config/openarm_v1.0/{openarm_bimanual.srdf,kinematics.yaml} /stage/'
    fi
    cp "$here"/*.py "$here"/*.rviz "$ctx/app/"
    docker build -t "$image" -f "$here/Dockerfile.local" "$ctx"
    ;;
  up)
    if docker container inspect "$name" >/dev/null 2>&1; then
      current=$(docker inspect -f '{{.Image}}' "$name")
      if [[ "$current" == "$(docker image inspect -f '{{.Id}}' "$image")" ]]; then docker start "$name"; exit; fi
      [[ "$(docker inspect -f '{{.Config.Image}}' "$name")" == "$image" ]] || exit 1
      if docker top "$name" -eo pid,args 2>/dev/null | grep -q '[r]epeat.py.*--execute'; then
        echo "Guarded run active; finish that run before updating the station" >&2; exit 1
      fi
      docker rm -f "$name" >/dev/null
    fi
    auth=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
    [[ -r "$auth" ]] || auth="$HOME/.Xauthority"
    [[ -r "$auth" ]] || { echo "No X11 cookie for $DISPLAY" >&2; exit 1; }
    out="$root/outputs/openarm_hardware"
    mkdir -p "$out/calibration"
    docker run -d --init --name "$name" --network host --cap-drop ALL --security-opt no-new-privileges \
      --hostname "$(hostname)" --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_HOME=/tmp/ros -e LAITING_FIXED_CHEST=1 \
      -e DISPLAY -e XAUTHORITY=/tmp/station.xauth -e QT_X11_NO_MITSHM=1 -e LIBGL_ALWAYS_SOFTWARE=1 \
      -e "ROS_DOMAIN_ID=${OPENARM_ROS_DOMAIN_ID:-88}" --log-opt max-size=10m --log-opt max-file=2 \
      -v "$auth:/tmp/station.xauth:ro" -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$out:/results" "$image"
    ;;
  ros2)
    shift; exec docker exec -i "$name" bash /opt/right-curl/app/env.sh ros2 "$@"
    ;;
  run)
    shift; exec docker exec -i "$name" bash /opt/right-curl/app/env.sh python3 "/opt/laiting/app/$1.py" "${@:2}"
    ;;
  *) echo "Usage: $0 {build|up|ros2 ...|run <program> ...}" >&2; exit 2 ;;
esac
