#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DESCRIPTION_DIR="$PROJECT_ROOT/third_party/openarm_description"
GENERATED_DIR="$DESCRIPTION_DIR/generated"
GENERATED_URDF="$GENERATED_DIR/openarm_v10_bimanual.urdf"
OPENARM_DESCRIPTION_COMMIT="1fba2cbc05001f05b4514120b70130b4ac06f409"

if [[ ! -d "$DESCRIPTION_DIR/.git" ]]; then
  mkdir -p "$PROJECT_ROOT/third_party"
  git clone https://github.com/enactic/openarm_description.git "$DESCRIPTION_DIR"
fi

git -C "$DESCRIPTION_DIR" fetch --depth 1 origin "$OPENARM_DESCRIPTION_COMMIT"
git -C "$DESCRIPTION_DIR" checkout --detach "$OPENARM_DESCRIPTION_COMMIT"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro is required (ROS 2 package: ros-humble-xacro)." >&2
  exit 1
fi

AMENT_STUB=$(mktemp -d)
cleanup() {
  unlink "$AMENT_STUB/share/openarm_description" 2>/dev/null || true
  rm -f "$AMENT_STUB/share/ament_index/resource_index/packages/openarm_description"
  rmdir "$AMENT_STUB/share/ament_index/resource_index/packages" 2>/dev/null || true
  rmdir "$AMENT_STUB/share/ament_index/resource_index" 2>/dev/null || true
  rmdir "$AMENT_STUB/share/ament_index" 2>/dev/null || true
  rmdir "$AMENT_STUB/share" 2>/dev/null || true
  rmdir "$AMENT_STUB" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$AMENT_STUB/share/ament_index/resource_index/packages" "$GENERATED_DIR"
touch "$AMENT_STUB/share/ament_index/resource_index/packages/openarm_description"
ln -s "$DESCRIPTION_DIR" "$AMENT_STUB/share/openarm_description"

AMENT_PREFIX_PATH="$AMENT_STUB${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}" \
  xacro \
  "$DESCRIPTION_DIR/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro" \
  bimanual:=true no_prefix:=false ros2_control:=false \
  -o "$GENERATED_URDF"

echo "Generated official OpenArm v1.0 bimanual body URDF:"
echo "  $GENERATED_URDF"
echo "Expected JointState names: openarm_left_joint1 ... openarm_left_joint7"
echo "                           openarm_right_joint1 ... openarm_right_joint7"
