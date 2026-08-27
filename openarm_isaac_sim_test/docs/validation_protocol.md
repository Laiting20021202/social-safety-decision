# Validation protocol

This protocol separates perception failures from planning/control failures and never treats
simulation as certification.

## Gate 1: deterministic stage

```bash
python3 -m pytest -q
/home/david/isaacsim/python.sh scripts/run_sim.py --headless --no-ros \
  --scenario no_obstacle --max-steps 240 --report-contacts \
  --capture-dir results/manual_stage_gate/camera
```

Pass conditions: six `CUBE_FINAL` lines match the seeded positions within 1 mm, all robot
states are finite, no unexpected `CONTACT_PAIR` appears, RGB is 640×480, and depth is finite
and expressed in metres. Inspect RGB visually; do not rely only on file existence.

## Gate 2: ROS RGB-D and TF

Launch the simulator with ROS and check:

```bash
ros2 topic hz /rgbd/color/image_raw
ros2 topic hz /rgbd/depth/image_raw
ros2 topic echo --once /rgbd/depth/camera_info
ros2 run tf2_ros tf2_echo world rgbd_color_optical_frame
```

Compare RGB/depth header stamps, verify `32FC1`, and open RGB, depth, point cloud, robot, and
Planning Scene in the supplied RViz config.

## Gate 3: control and sorting

Run `no_obstacle`. Confirm action servers, joint-state feedback, planning success, physical
gripper closure, one-cube lift without slip, correct release, and then all six cubes. A
`cube_placed` event alone is not success: evaluator ground-truth cube position must be inside
the matching bin. Preserve metrics, trajectory, screenshots, and bag.

## Gate 4: ground-truth safety

Run baseline, static, lateral crossing, sudden intrusion, withdraw/recover, fully blocked,
and repeated intrusion. Check Planning Scene obstacle updates, velocity scaling, trajectory
cancel, minimum clearance, latencies, false stops, replan, explicit recovery, and E-stop
latching. A fully blocked case should fail safely, not force task completion.

## Gate 5: perception-only safety

Before launch, audit subscriptions with `ros2 node info /openarm_safety_bridge`. It must have
no `/sim/ground_truth/*` input. Feed the external mask from RGB only, then compare evaluator
ground truth against perception timing. Repeat with camera timeout and depth frame drops.

Classification of failure:

- RGB/depth/mask absent or stale: perception/transport failure; fail-safe timeout expected.
- Obstacle cloud wrong but mask/depth present: perception/back-projection/TF failure.
- Correct cloud and Planning Scene, no plan: planning failure or genuinely blocked scene.
- Correct plan, action rejected or joint tracking poor: controller/bridge failure.
- Correct commanded motion, cube slips: grasp/contact/friction failure.

## Automated runner

Preview commands first:

```bash
python3 scripts/run_validation_suite.py --dry-run
```

Then run ground truth and perception separately. The perception suite is not considered
valid until the external mask publisher is running:

```bash
python3 scripts/run_validation_suite.py --mode ground_truth
python3 scripts/run_validation_suite.py --mode perception
```

Review every `metrics.json`; a runner process exit by itself is not evidence of a passed
scenario.

