#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=laiting-isaac-curl-station
out="$root/outputs/isaac_guarded"
model="${OPENARM_DESCRIPTION_DIR:-$root/.openarm-vendor/nuc-v10-description}"
case "${1:-up}" in
  build) docker build -t laiting-isaac-curl-control:humble -f "$here/Dockerfile" "$here" ;;
  up)
    if docker container inspect "$name" >/dev/null 2>&1; then
      if [[ $(docker inspect -f '{{.Image}}' "$name") != $(docker image inspect -f '{{.Id}}' laiting-isaac-curl-control:humble) ]]; then
        echo "Simulation container uses an older image. Finish any recording, then recreate $name." >&2
        exit 1
      fi
      docker start "$name"; exit
    fi
    auth=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
    [[ -r "$auth" ]] || auth="$HOME/.Xauthority"
    [[ -r "$auth" ]] || { echo "No readable X11 cookie for DISPLAY=$DISPLAY" >&2; exit 1; }
    docker run -d --init --name "$name" --network host --cap-drop ALL --security-opt no-new-privileges \
      --hostname "$(hostname)" --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_HOME=/tmp/ros \
      -e ROS_DOMAIN_ID=89 -e LAITING_CAMERA_PROFILE=isaac_fixed_openarm_v1 \
      -e DISPLAY -e XAUTHORITY=/tmp/isaac.xauth -e QT_X11_NO_MITSHM=1 -e LIBGL_ALWAYS_SOFTWARE=1 \
      --log-opt max-size=12m --log-opt max-file=2 -v "$auth:/tmp/isaac.xauth:ro" -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
      -v "$out:/results" -v "$here:/demo/isaac:ro" -v "$root/hardware/guarded_curl:/demo/guarded:ro" \
      -v "$root/hardware/right_curl/controllers.yaml:/demo/controllers.yaml:ro" \
      -v "$model:/model:ro" -v "$model:/opt/right-curl/share/openarm_description:ro" \
      laiting-isaac-curl-control:humble bash -c \
      'until [[ -s /results/calibration/isaac.json ]]; do sleep .5; done; export AMENT_PREFIX_PATH=/opt/isaac-hw:$AMENT_PREFIX_PATH LD_LIBRARY_PATH=/opt/isaac-hw/lib:$LD_LIBRARY_PATH; exec ros2 launch /demo/isaac/simulation.launch.py'
    ;;
  ros2) shift; exec docker exec -i "$name" bash /opt/right-curl/app/env.sh ros2 "$@" ;;
  run) shift; exec docker exec -i "$name" bash /opt/right-curl/app/env.sh python3 "/demo/guarded/$1.py" "${@:2}" ;;
  *) exit 2 ;;
esac
