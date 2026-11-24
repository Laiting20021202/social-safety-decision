# Environment report

Audit date: 2026-08-17 (Asia/Taipei). The workspace was empty and not a Git repository when
this project was created. Existing repositories were inspected read-only and their unrelated
working-tree changes were not modified.

## Host and simulator

| Component | Observed value | Compatibility decision |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-136 | Keep |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB | Suitable for one rendered environment |
| Driver | 550.163.01; CUDA compatibility 12.4 | Keep |
| Python | 3.10.12 | Matches ROS 2 Humble and Isaac Sim 4.5 |
| ROS 2 | Humble in `/opt/ros/humble` | Keep |
| ros2_control | 2.52.2 | Available |
| ros2_controllers | 2.50.2 | Available |
| MoveIt 2 Debian metapackage | Not installed; required system components are present | Use only `/opt/ros/humble` |
| Isaac Sim | `4.5.0-rc.36+release.19112...` at `/home/david/isaacsim` | Primary runtime |
| Isaac Lab | extension 0.46.3; repo describes `v2.2.1-77-gf52aa...-dirty` | Do not alter; not needed for this single scene |

The current official `openarm_isaac_lab` README targets newer Isaac Sim/Isaac Lab releases.
Forcing that stack onto this working Isaac Sim 4.5 host would be a high-risk upgrade. The
project therefore uses Isaac Sim's supported URDF importer with the official Enactic model,
while reusing official joint names, limits, and PD reference values. No Isaac Lab RL pipeline
is started.

The existing `/home/david/ws_moveit` overlay is incomplete (its setup references a missing
`robotiq_controllers` prefix) and ABI-incompatible with the system MoveIt 2.5.9 KDL library.
It was tested, produced an undefined-symbol error, and is deliberately not used. The launch
file loads the official OpenArm SRDF/kinematics/controllers directly and supplies a
Humble-compatible scalar OMPL adapter parameter.

## OpenArm sources

The existing official description checkout is:

```text
/home/david/Desktop/laiting/itri/3d_safety_decision/
  realtime_3d_safety_decision/third_party/openarm_description
revision: 1fba2cbc05001f05b4514120b70130b4ac06f409
```

It contains v1.0/v2.0 xacro, meshes, bimanual ros2_control descriptions, and a user-created
untracked `generated/` directory. The checkout was not changed. `dependencies.repos` also
pins official `openarm_ros2` revision
`4e837e1d0dae692ff67b560b69d8d281d7a8d4ed`, which provides the bimanual SRDF, kinematics,
joint limits, MoveIt controllers, and `hands_up` state.

The generated simulator cache records its source in `assets/openarm_cache/source.json` and
is ignored by Git. A user can instead provide an official USD through `OPENARM_USD_PATH`.

## Existing perception repository

Inspected checkout:

```text
/home/david/Desktop/laiting/itri/social-safety-decision
branch: complete-gpu-gui-system
revision: f31b...
working tree: clean at audit time
```

Findings:

- It has RGB/depth, YOLO/SAM services, point-cloud/BEV processing, tracking, and danger-zone
  concepts useful for obstacle extraction.
- Its current primary workflow is offline/dataset playback and AMR-oriented 2D/BEV safety.
- `services/ros2_adapter/__init__.py` is a placeholder.
- It does not currently expose a live standard ROS RGB-D input and tracked 3-D OpenArm
  obstacle output.
- Its forward-corridor/local-planner concepts and `Twist` commands are not used to command
  OpenArm.

For this reason, perception mode uses an independent mask/depth adapter. The external
pipeline must publish `/perception/hand_mask`; ground truth is forbidden as safety input.

## Actual simulator checks performed

Commands included:

```bash
/home/david/isaacsim/python.sh scripts/create_scene.py --headless --warmup-frames 30
/home/david/isaacsim/python.sh scripts/run_sim.py --headless --no-ros \
  --max-steps 240 --report-contacts --capture-dir results/isaac_smoke_stable/camera
```

Observed:

- Scene build and USD export completed using the official OpenArm URDF.
- 18 controllable DOFs were found.
- Six cubes remained at their deterministic table positions after 240 steps.
- No PhysX contact pair was reported in the settled `hands_up` scene.
- RGB was 640×480; depth was float32, 480×640, all pixels valid in the smoke capture,
  approximately 0.575–2.814 m.
- `enabled_self_collisions=True` and global gravity remained enabled.

The initial instability was traced to Isaac Sim 4.5 writing a requested robot pose onto the
imported fixed-joint prim. Moving the parent reference Xform fixed the entire robot being
embedded in the ground/table. A remaining valid-pose overlap between the simplified v1.0
`link5` and `link7` collision meshes is treated as a wrist-enclosure adjacent filter. This is
explicit and narrowly scoped; self-collision is not globally disabled.

Under gravity and official effort limits the official `hands_up` pose is compliant and sags;
it is collision-free, but it is not a rigidly locked pose. Full MoveIt/grasp validation is
still required before claiming task success.
