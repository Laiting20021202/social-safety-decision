# OpenArm simulation

This ROS 2 package owns the simulated scene and assets. The Docker Isaac service runs
`scripts/run_sim.py` to load the YAML configuration, create the PhysX scene, and start
`openarm_isaac_bridge` for ROS 2 sensors, joint feedback and controller actions.

| Path | Role |
|---|---|
| `config/` | Robot, camera, workcell, hand scenarios and evaluation settings |
| `openarm_sim/isaac_scene.py` | OpenArm articulation, cameras, props and animated obstacle |
| `openarm_sim/hand_runtime.py` | Scripted and manual obstacle motion |
| `assets/`, `worlds/` | Scene assets and world generation output |
| `scripts/` | Asset generation, simulator launch, validation and legacy Gazebo utilities |

[Workspace setup](../../README.md) · [ROS 2 interfaces](../../docs/ARCHITECTURE.md) · [Validation protocol](docs/validation_protocol.md) · [Historical Gazebo operations](https://github.com/Laiting20021202/social-safety-decision/blob/02563df2da4494cb90703f203561d978eaed6d70/openarm_isaac_sim_test/README.md)
