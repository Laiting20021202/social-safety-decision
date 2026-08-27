# OpenArm URDF in the metric point-cloud GUI

The `koch_lan` profile renders the official Enactic OpenArm v1.0 **bimanual**
URDF: central body, left seven-axis arm, right seven-axis arm, and both
grippers. It shares one metre-based Viser scene with the reconstructed work
environment and tracked obstacles. The mesh source is pinned by
`scripts/setup_openarm.sh`; generated and upstream assets stay under the
ignored `third_party/openarm_description` directory.

## Setup

```bash
bash scripts/setup_openarm.sh
```

This expands the upstream v1.0 xacro with `bimanual:=true`,
`no_prefix:=false`, and `ros2_control:=false`. The resulting actuated joints
are:

```text
openarm_left_joint1 ... openarm_left_joint7
openarm_right_joint1 ... openarm_right_joint7
openarm_left_finger_joint1
openarm_right_finger_joint1
```

Each mirrored second finger follows its first finger through the URDF mimic
joint. A complete live body pose requires all 14 arm joints; either gripper can
remain at its last value when its finger joint is absent.

## Live JointState contract

Default topic: `/joint_states` (`sensor_msgs/msg/JointState`). Names and
positions must have equal lengths and positions are radians, except the
prismatic finger joint, which is metres. Exact official names are preferred.
The GUI accepts exact official names and side-preserving aliases such as
`left_joint1` and `right_joint1`. An ambiguous `joint1` is deliberately not
copied into both arms.

Verify the hardware stream:

```bash
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=42 ros2 topic echo /joint_states --once
```

The OpenArm panel distinguishes three states:

- `URDF READY / WAITING`: official geometry is visible at zero pose, no live
  JointState has arrived.
- `JOINT LIVE`: a sample arrived within `openarm.stale_after_s` and the panel
  shows how many of the 14 left/right arm joints matched.
- `JOINT STALE`: the last pose remains visible but its age is shown; it is not
  represented as a fresh robot state.

The collapsed joint-inspection folder contains every URDF joint value. Disable
**追蹤 ROS JointState** to move the manual test sliders; re-enable it to return
to the most recently received robot pose.

## Spatial coordinate contract

The live 8 × 8 cm AprilTag is the spatial and scale anchor. Its four 3D sides
and two diagonals set the global point-cloud scale. The OpenArm body base is:

```text
base = AprilTag center + openarm.base_from_apriltag_xyz
```

The configured offset `[0.0, -0.50, 0.0]` puts the full body at the requested
central red point, exactly 0.500 m below the tag in the work-plane view.
`openarm.base_rpy_deg` supplies the remaining body orientation. Coordinates are:

```text
x = right across the table
y = forward along the table
z = height above the fitted table
```

With Metric BEV enabled, the RANSAC-fitted table is `z=0`, so the URDF base,
orthorectified cloud, obstacle boxes, and 5 cm grid are directly comparable in
metres. In raw-camera view, the same pose is transformed back through the
fitted plane basis and point-cloud centering; the geometry does not get an
unrelated display-only rotation.

The GUI draws the detected 8 cm tag outline in orange and a labelled 0.500 m
Tag→OpenArm-body line. Camera height 0.50 m and downward angle 36 degrees remain
survey references; they no longer determine point-cloud scale. Fine-tune the
Tag→Base XYZ and body RPY only after verifying the printed tag's black-square
side is physically 80 mm.

Green and cyan left/right TCP markers are computed from URDF forward
kinematics. Each obstacle is connected to the nearer TCP, and the ROS JSON
includes both named TCP centers plus `nearest_arm_point`. The BEV also keeps
tracked obstacle wireframe volumes visible instead of hiding them with the
raw-camera scene layer.

## Current safety boundary

URDF visualization and FK are integrated, but the ROS obstacle cloud does not
yet subtract robot collision geometry. Keep the existing controller timeout,
STOP behavior, and physical safety controls. Enable link self-filtering only
after the camera-to-base extrinsic and link radii/collision meshes have been
measured and validated on the real cell.
