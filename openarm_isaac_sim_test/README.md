# OpenArm dynamic avoidance and control layer

This directory is the control-only snapshot used by
`layer/dynamic-avoidance-control`.  It intentionally excludes Gazebo/Isaac
assets and launch files; those are added by `layer/simulation-validation` and
`main`.

## Data flow

```text
/perception/obstacles + /joint_states + requested TCP target
  -> openarm_safety_bridge
  -> openarm_dynamic_avoidance
  -> openarm_sorting_task / MoveIt
  -> FollowJointTrajectory-compatible OpenArm controller
```

`openarm_safety_bridge` maintains the `SAFE`, `WARNING`, `PAUSE`, `REPLAN`,
`RECOVER` and `EMERGENCY_STOP` state machine.  Perception mode subscribes only
to perception topics; simulator ground truth is forbidden by the interface
contract.  `openarm_dynamic_avoidance` updates collision geometry, motion
guards and trajectory scaling.  `openarm_sorting_task` owns pose goals and the
hold/vertical-evade/restore behavior.

MoveIt/OMPL is the current planning baseline.  This branch does not contain a
trained MπNets or other learned motion policy and must not be described as one.

## Setup and unit tests

```bash
cd openarm_isaac_sim_test
python3 -m pip install --user -e '.[dev]'
source /opt/ros/humble/setup.bash
vcs import ros2_ws/src < dependencies.repos
colcon build --base-paths ros2_ws/src --symlink-install \
  --packages-select openarm_dynamic_avoidance openarm_safety_bridge \
                    openarm_sorting_task
source install/setup.bash
python3 -m pytest -q
```

For the one-command Gazebo demo, camera pipeline, hand sweep and validation
runner, switch to `main` and follow this file at the same path; the full branch
contains the remaining simulator packages and commands.

## Important control behavior

- TCP goals remain active after arrival; Home is an explicit command.
- A nearby moving hand cancels an obsolete trajectory and requests a
  collision-checked escape/replan instead of treating a soft warning as a
  permanent stop.
- During target hold the reflex lifts the affected TCP.  When the obstacle is
  clear, it restores the saved target posture rather than returning Home.
- `EMERGENCY_STOP` remains latched until an explicit reset.
- Ground-truth distance may be used by evaluators, never by the perception
  planner.
