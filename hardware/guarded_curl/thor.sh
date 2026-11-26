#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
image=${THOR_IMAGE:-laiting-openarm-mast3r:jazzy}
name=laiting-openarm-thor
case "${1:-up}" in
  build)
    existing=${THOR_MAST3R_ADAPTER:-$HOME/mast3r_openvins/master_openarm}
    ctx=$(mktemp -d)
    trap 'rm -rf -- "$ctx"' EXIT
    cp "$existing"/{thor_live,thor_geometry,thor_tracker}.py "$here"/{frames,thor_cloud}.py "$ctx/"
    cp "$here/../right_curl/fastdds.xml" "$ctx/"
    docker build -t "$image" -f "$here/Dockerfile.thor" "$ctx"
    ;;
  up)
    if docker container inspect "$name" >/dev/null 2>&1; then
      if [[ "$(docker inspect -f '{{.Image}}' "$name")" == "$(docker image inspect -f '{{.Id}}' "$image")" ]]; then
        docker start "$name"; exit
      fi
      [[ "$(docker inspect -f '{{.Config.Image}}' "$name")" == "$image" ]] || exit 1
      docker rm -f "$name" >/dev/null
    fi
    docker run -d --init --name "$name" --network host --runtime=nvidia --cpus=6 --memory=16g \
      --shm-size=512m --cap-drop ALL --security-opt no-new-privileges --log-opt max-size=8m --log-opt max-file=2 \
      -e "ROS_DOMAIN_ID=${OPENARM_ROS_DOMAIN_ID:-88}" -e LAITING_FIXED_CHEST=1 "$image"
    ;;
  *) exit 2 ;;
esac
