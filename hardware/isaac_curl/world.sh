#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$root/outputs/isaac_guarded"
model="${OPENARM_DESCRIPTION_DIR:-$root/.openarm-vendor/nuc-v10-description}"
hand="${OPENARM_HAND_DIR:-$here/assets/libhand}"
[[ -s "$hand/hand.obj" || -s "$hand/hand.obj.gz" ]] || { echo "Missing LibHand geometry in $hand" >&2; exit 1; }
for item in "$hand/hand_texture.png" "$model/urdf/robot/v10.urdf.xacro"; do [[ -f "$item" ]] || { echo "Missing asset: $item" >&2; exit 1; }; done
(cd "$model" && sha256sum --check --quiet "$here/../right_curl/site-v10.sha256")
mkdir -p "$out"/{assets,calibration,usd}
if [[ ! -s "$out/assets/openarm.urdf" ]]; then
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$out/assets:/out" \
    -v "$model:/opt/right-curl/share/openarm_description:ro" laiting-openarm-safety:humble bash -c \
    'xacro /opt/right-curl/share/openarm_description/urdf/robot/v10.urdf.xacro bimanual:=true hand:=true ros2_control:=false | sed "s|package://openarm_description/|/model/|g" > /out/openarm.urdf'
fi
name=laiting-isaac-fixed-world
if docker container inspect "$name" >/dev/null 2>&1; then docker start "$name"; exit; fi
[[ ${ACCEPT_EULA:-} == Y ]] || { echo 'Set ACCEPT_EULA=Y after accepting the NVIDIA Isaac Sim terms' >&2; exit 1; }
docker run -d --init --name "$name" --gpus all --network host --shm-size=1g \
  --log-opt max-size=14m --log-opt max-file=2 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ROS_DOMAIN_ID=89 -e ROS_LOCALHOST_ONLY=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/demo/fastdds.xml \
  -v "$here:/demo/app:ro" -v "$root/hardware/right_curl/fastdds.xml:/demo/fastdds.xml:ro" \
  -v "$model:/model:ro" -v "$hand:/hand:ro" -v "$out:/run-data" \
  -v "$out/usd:/opt/social-safety-decision/src/openarm_sim/assets/openarm_cache" \
  -v laiting-isaac-rtx-cache:/root/.cache -v laiting-isaac-kit-cache:/isaac-sim/kit/cache \
  social-safety-decision-isaac:4.5.0 bash -c \
  'export PYTHONPATH=/opt/social-safety-decision/src/openarm_sim:$PYTHONPATH; exec /isaac-sim/python.sh /demo/app/isaac_world.py'
