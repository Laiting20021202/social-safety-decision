# Repository handoff verification — 2026-09-22

Base commit: `7963c4b` on `main`; this report accompanies the September Isaac
LAN source and documentation handoff. The final handoff commit is the commit
containing this file (use `git log -1 --format=%H -- results/repo_handoff_2026-09-22.md`).

## Checks actually run

From the repository root:

```bash
.venv/bin/python -m pytest -q
```

Result: **310 passed, 1 skipped in 5.75s**. The skipped case is
`tests/test_pipeline_smoke.py:14`: neural-model/CUDA smoke requires
`SAFETY_SMOKE_VIDEO`. These tests include the nine new joint-jog state cases;
they do not exercise a live Isaac controller.

From `openarm_isaac_sim_test/`:

```bash
../.venv/bin/python -m pytest -q
```

Result: **105 passed in 0.55s**. These are source/config/math/control tests,
not a simulator runtime or hardware acceptance test.

Reproduce both suites plus shell syntax, Python compilation and whitespace checks:

```bash
bash scripts/check_repo.sh
```

Final run of that script passed: root **310 passed, 1 skipped in 5.47s**;
simulation **105 passed in 0.48s**. Shell syntax, Python compilation and
`git diff --check` passed. Local file links in 12 handoff/reference documents
and the annotated historical readiness JSON were also checked successfully.

## Read-only host audit

Command: after sourcing `/opt/ros/humble/setup.bash`, run
`.venv/bin/python openarm_isaac_sim_test/scripts/validate_environment.py --json`.

| Item | Observation |
|---|---|
| Python | 3.10.12 |
| ROS | Humble; `ros2`, `colcon`, `vcs`, `xacro` found |
| GPU | RTX 4060 Ti, 16380 MiB, driver 550.163.01 |
| Isaac | `/home/david/isaacsim`, version `4.5.0-rc.36+release.19112.f59b3005.gl` |
| OpenArm description | `third_party/openarm_description`, revision `1fba2cbc05001f05b4514120b70130b4ac06f409` |
| `moveit_ros_move_group` | Found at `/opt/ros/humble` |
| `moveit_simple_controller_manager` | Not found in the system ROS environment; runtime prerequisite remains |
| `ros-humble-moveit` metapackage | Not installed; individual MoveIt components may still exist |
| `openarm_isaac_sim_test/ros2_ws/src/external` | Dependencies not present at this path; follow dependency setup or configure existing checkouts |
| GUI services | `isaac-lan-gui.service` and `realtime-safety-3d.service`: inactive |
| TCP 8080 | No listener at inspection time |

No model download, full colcon rebuild, GUI startup, remote ROS discovery,
Isaac/Gazebo execution, motion command or physical-arm test was performed.
The 9/17 LAN observations are historical; their later unexpected-movement
report remains unresolved. See [the handoff checklist](../docs/ISAAC_SIM_HANDOFF.md)
for blockers and the required runtime acceptance evidence.
