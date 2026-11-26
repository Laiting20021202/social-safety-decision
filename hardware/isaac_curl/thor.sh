#!/usr/bin/env bash
set -euo pipefail
name=laiting-isaac-thor
image=laiting-isaac-mast3r:jazzy
case "${1:-up}" in
  build) export THOR_IMAGE="$image"; exec bash "$(dirname "$0")/../guarded_curl/thor.sh" build ;;
  up) ;;
  *) echo "Usage: $0 {build|up}" >&2; exit 2 ;;
esac
if docker container inspect "$name" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.Image}}' "$name")" == "$(docker image inspect -f '{{.Id}}' "$image")" ]]; then docker start "$name"; exit; fi
  [[ "$(docker inspect -f '{{.Config.Image}}' "$name")" == "$image" ]] || exit 1
  docker rm -f "$name" >/dev/null
fi
docker run -d --init --name "$name" --network host --runtime=nvidia --shm-size=512m --cpus=6 --memory=16g \
  --cap-drop ALL --security-opt no-new-privileges --log-opt max-size=8m --log-opt max-file=2 \
  -e ROS_DOMAIN_ID=89 -e LAITING_FIXED_CHEST=1 "$image"
