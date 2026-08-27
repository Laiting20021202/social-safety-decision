# OpenArm demo handoff status

Last updated: 2026-08-27

## Current scope

The active end-to-end path is Gazebo Classic + RViz + ROS 2 Humble.  Gazebo
publishes generic synchronized RGB-D; `realtime_3d_safety_decision` rebuilds
the current-frame environment and hand clouds; MoveIt plus the dynamic safety
layer sends standard trajectories to the bimanual OpenArm controllers.

The two Gazebo cubes are TCP targets, not grasp objects.  A commanded arm stays
at its target until **HOME** is pressed.  If a tracked hand approaches, it
executes a collision-checked escape and restores the saved target after the
obstacle clears.  Perception mode does not give simulator ground truth to the
planner.

## Reproduce

```bash
cd openarm_isaac_sim_test
./scripts/start_gazebo_demo.sh
# Wait for: OPENARM GAZEBO DEMO READY
./scripts/run_dynamic_avoidance_test.sh
```

The web GUI is normally served at `http://<host-ip>:8080/`.  Use Gazebo's
Translate/Rotate tools for the camera, target cubes and hand.  Do not move the
OpenArm fixed base without regenerating the robot/planning-frame configuration.

## Verification recorded before handoff

| Check | Result |
|---|---|
| Perception/config/interface pytest suite | 301 passed, 1 skipped |
| Control-layer pytest suite | 56 passed |
| Complete simulation source pytest suite | 105 passed |
| RGB/depth/CameraInfo current-frame back-projection | Passed on the audited Gazebo host |
| Non-empty model-confirmed moving-hand cloud | Passed; visibility and hand-gate confidence dependent |
| MoveIt dual TCP goals through FollowJointTrajectory | Passed on the audited Gazebo host |
| Hold, evade, then restore the saved target | Implemented and covered by deterministic tests |
| Isaac visible runtime after this packaging pass | Not rerun; Gazebo is the active runtime |
| Physical OpenArm safety validation | Not performed; simulation is not safety certification |

## Known limitations

- The retained COCO YOLO checkpoint has no human-hand class.  MediaPipe supplies
  hand semantics and EdgeTAM refines/tracks the mask; model setup scripts fetch
  weights that are intentionally excluded from Git.
- Hand visibility, RGB-D occlusion, camera calibration and self-filter alignment
  directly affect the obstacle cloud.  A stale/missing cloud must remain a
  visible fail-safe condition, not silently switch to simulator ground truth.
- The current dynamic layer is MoveIt/OMPL plus online safety/replanning.  It is
  not a trained MπNets policy and must not be reported as one.
- Generated worlds, USD, ROS build products, bags, model weights, result folders
  and vendored repositories are excluded.  Recreate them with the committed
  setup and scene scripts.

## Ownership

- Perception, RGB-D and the 8080 viewer: `layer/perception-detection`.
- Planning, control and hold/evade/restore: `layer/dynamic-avoidance-control`.
- Gazebo/Isaac scenes, launch and evaluators: `layer/simulation-validation`.
- `main`: reviewed integrated handoff snapshot.

Update this file with the exact command, result artifact and remaining issue
whenever an acceptance test changes.
