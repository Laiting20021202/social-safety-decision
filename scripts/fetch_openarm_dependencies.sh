#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p .openarm-vendor src
vcs import --shallow .openarm-vendor < openarm.repos

# Only the model and MoveIt config participate in this workspace's build.
for entry in \
  'openarm_description:openarm_description' \
  'openarm_bimanual_moveit_config:openarm_ros2/openarm_bimanual_moveit_config'; do
  slot="src/${entry%%:*}"
  vendor_path="../.openarm-vendor/${entry#*:}"
  if [[ ! -f "src/$vendor_path/package.xml" ]]; then
    echo "Missing ROS package in $vendor_path" >&2
    exit 1
  fi
  if [[ -L "$slot" && "$(readlink -- "$slot")" == "$vendor_path" ]]; then
    continue
  fi
  ln -sT -- "$vendor_path" "$slot"
done
