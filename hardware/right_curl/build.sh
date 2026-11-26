#!/usr/bin/env bash
set -euo pipefail
role=${1:?Use robot on the NUC or viewer on the RViz workstation}
case "$role" in
  robot) default_base=openarm-dev:latest ;;
  viewer) default_base=osrf/ros:humble-desktop ;;
  *) echo "Expected robot or viewer" >&2; exit 2 ;;
esac
here=$(cd -- "$(dirname -- "$0")" && pwd)
lab_src=${OPENARM_LAB_SRC:-$HOME/jeff/openarm_dev/ros2_ws/src}
description=${OPENARM_DESCRIPTION_DIR:-$lab_src/openarm_description}
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/description" "$stage/app"
cp -a "$description"/{config,urdf,meshes,package.xml,LICENSE.txt} "$stage/description/"
cp -a "$here"/{Dockerfile,fastdds.xml} "$stage/"
cp -a "$here"/{CMakeLists.txt,package.xml,plugin.xml,*.cpp,*.py,*.yaml,*.rviz,env.sh} "$stage/app/"
if [[ $role == robot ]]; then
  cp -a "${OPENARM_CAN_DIR:-$lab_src/openarm_can}" "$stage/can"
  rm -rf -- "$stage/can/.git"
  mkdir "$stage/camera"
  cp "$here/../../docker/fastdds.xml" "$stage/camera-source.xml"
  rs_root=${OPENARM_REALSENSE_DIR:-$lab_src/realsense-ros}
  cp -a "$rs_root"/{realsense2_camera,realsense2_camera_msgs} "$stage/camera/"
  patch --no-backup-if-mismatch -d "$stage/camera" -p1 < "$here/d415-options.patch"
fi
docker build --network none --target "$role" --build-arg "BASE_IMAGE=${BASE_IMAGE:-$default_base}" \
  -t "laiting-openarm-$role:humble" "$stage"
