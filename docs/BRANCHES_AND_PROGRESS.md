# Branches, ownership and current progress

Last updated: 2026-09-22

> This page records the September handoff. See the [workspace README](../README.md) for the current layout and launch commands.

## Current handoff: Isaac Sim migration

The current `main` handoff adds the September Isaac LAN viewer work and a
[step-by-step Isaac migration checklist](ISAAC_SIM_HANDOFF.md). The full
Isaac perception-to-avoidance loop is **not yet accepted**. Gazebo is the
previous integration baseline; this week's target runtime is Isaac Sim.
The LAN viewer is RGB-only with unverified metric scale; its experimental
joint jog remains disabled by default following the September 17 report of
unexpected movement. Root tests: 310 passed, 1 skipped; simulation tests:
105 passed on September 22. See the [verification report](../results/repo_handoff_2026-09-22.md).

This handoff is committed on `main`. The cumulative layer branches below are
historical architecture snapshots and have not been advanced by this update.
Start new Isaac work from the current `main`, then integrate tested changes;
cloning `layer/simulation-validation` alone will miss this handoff.

## Branch policy

The active branches are cumulative:

```text
layer/perception-detection
  └─ layer/dynamic-avoidance-control
       └─ layer/simulation-validation
            └─ main
```

Make perception-only changes on `layer/perception-detection`.  Merge that
branch upward after its camera/topic tests pass.  Make planner and controller
changes on `layer/dynamic-avoidance-control`, then merge upward after unit and
MoveIt tests pass.  Gazebo, Isaac, launch and evaluator changes belong on
`layer/simulation-validation`.  `main` is a handoff/demo snapshot, not an
experimental branch.

The old `2d_version` and `3d_version` branches are retained to preserve prior
work.  Do not force-push or delete them.

## Layer 1 — Perception & Detection

Primary paths:

```text
realtime_safety/pipeline/       RGB-D, masks, tracking and 3D obstacles
realtime_safety/edgetam_tracker EdgeTAM point-cloud tracking
realtime_safety/ros2_bridge/    ROS 2 image/cloud/control adapters
realtime_safety/gui/            8080 Viser dashboard
config/ and configs/            ROS and application profiles
launch/                         perception launch descriptions
tests/                          deterministic perception/interface tests
```

Implemented:

- Generic ROS 2 RGB-D input; no RealSense-specific dependency.
- Current-frame metric depth back-projection and environment PointCloud2.
- Selectable legacy YOLO/MediaPipe and EdgeTAM hand pipelines.
- Robot self-filter, dynamic-obstacle cloud, temporal tracking and ROS cloud
  source mux.
- Live 8080 RGB, point cloud, OpenArm state and pipeline diagnostics.

Known limitations:

- The checked-in COCO YOLO weights do not contain a human-hand class;
  MediaPipe supplies hand semantics in the retained path.
- Model weights and third-party model repositories must be installed with the
  provided scripts.
- Simulated performance is not a hardware safety certification.

## Layer 2 — Dynamic Avoidance & Control

Added on `layer/dynamic-avoidance-control` under
`openarm_isaac_sim_test/ros2_ws/src/`:

```text
openarm_dynamic_avoidance/  obstacle motion, safety commands and replanning
openarm_safety_bridge/      perception/ground-truth isolation and safety FSM
openarm_sorting_task/       MoveIt pose goals and hold/evade/restore behavior
```

The interface is:

```text
environment/dynamic obstacle clouds + /joint_states + target
  -> dynamic safety and MoveIt planning
  -> trajectory_msgs/JointTrajectory / FollowJointTrajectory
  -> OpenArm controller
```

MoveIt/OMPL remains the planning baseline.  It is not labelled as a trained
MπNets or motion-policy-network model.

## Layer 3 — Simulation & Validation

`layer/simulation-validation` adds the complete, source-only
`openarm_isaac_sim_test/` tree.  Generated USD, ROS build products, result
directories and vendored repositories remain ignored.

Previous integrated runtime: Gazebo Classic + RViz + MoveIt + ROS 2 Humble.
The active development target is now Isaac Sim. On September 22 both local
GUI services were inactive; no live demo is asserted by this document.
The Gazebo one-click launcher and dynamic-hand test remain regression tools.
Use `docs/ISAAC_SIM_HANDOFF.md` for the current Isaac work, and
`openarm_isaac_sim_test/README.md` for historical runtime commands.

## Handoff checklist

Before pushing a layer upward:

1. Confirm `git status` contains no credentials, caches, model weights or
   generated build products.
2. Run the layer's pytest suite and ROS import/build smoke tests.
3. Record what was actually tested and what remains unverified.
4. Keep perception and simulator-ground-truth topics isolated.
5. Update this document and the relevant package README when an interface or
   launch command changes.
