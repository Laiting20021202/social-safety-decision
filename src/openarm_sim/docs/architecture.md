# Architecture

The simulator owns environment geometry, dynamics, sensing, deterministic ground truth, and
evaluation. MoveIt owns task motion planning. The safety bridge is replaceable and owns the
policy decision; it can later be replaced by MoveIt Servo, CBF, or another planner without
rewriting the stage.

```text
                            ┌──────────────── evaluator only ────────────────┐
Gazebo/Isaac ─ RGB-D topics ┼─ MediaPipe + EdgeTAM ─ obstacle cloud ┐       │
      │                     │                                       │       │
      ├─ GT hand namespace ─┼────────────────── GT mode only ───────┤       │
      │                     └────────────────────────────────────────┼───────┘
      │                                                              ▼
      └─ joints/actions ◀── trajectory bridge ◀── MoveIt ◀── safety bridge
                                  │                        │
                                  └── /joint_states ───────┘
```

## Mode isolation

`openarm_sim.contracts` defines the only permitted safety subscriptions:

- Ground-truth: `/sim/ground_truth/hand_collision` and
  `/sim/ground_truth/min_distance`.
- Perception: `/perception/obstacles` only.

The evaluator may subscribe to ground truth in either mode because it scores actual
clearance and cube position; it has no command publisher. Tests fail if a perception safety
subscription includes `/sim/ground_truth/*`.

## ROS contracts

The active Gazebo bridge publishes `/clock`, `/joint_states`, synchronized generic RGB-D,
CameraInfo, PointCloud2, camera TF, cube ground truth, and hand ground truth. It hosts the
four action names from the official MoveIt controller YAML:

```text
/left_joint_trajectory_controller/follow_joint_trajectory
/right_joint_trajectory_controller/follow_joint_trajectory
/left_gripper_controller/gripper_cmd
/right_gripper_controller/gripper_cmd
```

The trajectory bridge interpolates accepted goals against simulator time and reports joint
feedback. `PAUSE`/`EMERGENCY_STOP` freezes targets. MoveIt receives the same joint names and
base transform as the Isaac model.

## Planning and safety

The pose-goal node sends standard `moveit_msgs/action/MoveGroup` goals. Ground-truth mode inserts
the exact box/capsule proxy approximation. Perception mode robustly bounds the 2–98%
quantiles of the tracked point cloud and inserts that box. The safety policy uses hysteresis,
freshness timeout, a latched E-stop, velocity scaling, trajectory cancellation, replan, and
explicit recovery.

The Gazebo implementation uses discrete Planning Scene updates rather than an embedded
simulator avoidance rule. A non-emergency intrusion first cancels the obsolete trajectory;
after the padded perception obstacle is committed, MoveIt computes a new spatial path and
the external dynamic layer executes it at reduced speed. Continued obstacle displacement
triggers throttled replanning. Emergency clearance still latches a hard stop. This keeps
the safety system externally replaceable and does not label OMPL replanning as MπNets.

## Assets and collision policy

OpenArm meshes/URDF are not copied into this repository. The resolver uses a pinned Enactic
checkout or `OPENARM_USD_PATH`. Gazebo renders a locally converted, textured LibHand asset;
hidden palm/forearm proxies provide conservative collision geometry. See
`assets/hand/README.md`.
