# Tidybot retreat implementation checks — 2026-09-30

Requested behavior: proximity within 1 m triggers motion opposite the obstacle,
with 2 m travel measured from base feedback. Implemented as an independent
simulation-only ROS adapter and a pure Python controller. Existing unrelated
worktree changes and the camera GUI were left intact.

## Tests performed

- `.venv/bin/python -m pytest -q tests/test_tidybot_retreat.py`: **24 passed**.
  Includes rotated frames, left/right obstacles, a kinematic feedback loop to
  2 m within 3 cm, latched completion, stale/replayed input, invalid data,
  clock reset, pose jumps, stall, explicit reset, density filtering and
  PointCloud2 row padding/endianness.
- `source /opt/ros/humble/setup.bash` followed by
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp RUN_TIDYBOT_ROS_TEST=1 .venv/bin/python -m pytest -q tests/test_tidybot_retreat_ros.py`:
  **1 passed in 18.35s**. A fake kinematic receiver on **domain 179,
  localhost-only** consumed real ROS Twist messages, published pose/cloud
  feedback, reached the negative-X 2 m target within tolerance, and received
  zero velocity at completion. It did not connect to domain 0 / live Isaac.
- Python compilation and launch-script shell syntax checks passed.
- `--execute` with the checked-in unverified calibration is rejected before
  ROS initialization. Verification gates intentionally remain false.

## Live read-only checks

A 12-second bounded dry-run on domain 0 consumed live
`/realtime_safety/pointcloud` and `/tidybot/joint_states`. It reached `IDLE`,
reported fresh data and base feedback, and had no supported obstacle inside
the configured example ROI. This is not evidence of correct geometry:
camera extrinsics and metric scale are placeholders, and no intruding-object
accuracy test was run.

The actual `/tidybot/cmd_vel` subscriber was discovered as
`_World_TidybotBaseControl_SubscribeTwist`. No motion command was sent to live Isaac.

## Isaac control audit supplied after the test

The inspected `/World/TidybotBaseControl` graph confirms body-frame `linear.x`
(forward), `linear.y` (left), and `angular.z` (CCW), conversion through the current
`joint_th`, and position-target integration for `joint_x`, `joint_y`, and `joint_th`.
Its receiver stops integrating and holds the current pose after 0.5 s without a new
message. The front camera pose relative to the rotating base is
`(0.1000000, 0.0102028, 1.3149987) m`, pitched downward 20 degrees; this transform is
now recorded in `configs/tidybot_retreat.yaml`.

This does not validate the metric scale of `/realtime_safety/pointcloud`, because that
cloud is reconstructed from RGB-estimated depth rather than Isaac metric depth. Live
execution therefore remains blocked on metric-depth calibration and a visual check of
the complete 2 m retreat corridor.

## Outstanding acceptance

Calibrate depth scale or use metric RGB-D, and confirm an unobstructed retreat
corridor. Then test real simulated travel, rotated base, cancellation,
camera/joint interruptions and receiver timeout. The front-camera reflex does
not check the complete retreat corridor and is not a collision-free planner.

Commands and limitations: [Tidybot retreat guide](../docs/TIDYBOT_RETREAT.md).

## Live Isaac lateral retreat acceptance

After the receiver audit, two controlled domain-0 trials used a dedicated synthetic
metric obstacle topic in the base frame while retaining the real Isaac
`/tidybot/joint_states` feedback and `/tidybot/cmd_vel` subscriber:

- Left obstacle at approximately `(0.011, +0.704) m`: the base retreated right from
  `(-0.0070, -0.6524)` toward `(-0.0366, -2.6522)`. It entered `DONE` at
  `(-0.0362, -2.6256)`, 0.0266 m from the 2 m target.
- Right obstacle at approximately `(0.010, -0.700) m`: the base retreated left from
  `(-0.0363, -2.6287)` toward `(-0.0656, -0.6300)`. It entered `DONE` at
  `(-0.0651, -0.6596)`, 0.0296 m from the 2 m target.

Both trials sent zero velocity after completion. A preceding partial trial stopped at
about 0.65 m when its synthetic cloud expired; the controller correctly latched
`FAULT: stale point cloud or joint feedback`, and Isaac stopped. The adapter now gives
DDS graph discovery three seconds before enforcing sole-publisher arbitration, without
publishing motion during that grace period.

After cleanup, `/tidybot/cmd_vel` had zero publishers and exactly the intended Isaac
subscriber. Final feedback was approximately `joint_x=-0.0652`, `joint_y=-0.6533`,
`joint_th=-0.0001`, with the base stopped. These trials validate the command and
feedback loop in both lateral directions; they do not validate the scale of the live
RGB-estimated `/realtime_safety/pointcloud`.

## Live RGB-estimated cloud trigger

After restoring the remote camera, joint-state publisher, and base subscriber, the
uncalibrated simulation mode triggered from the live `/realtime_safety/pointcloud`.
The latched target was `(-1.8655, -0.7202) m`; final feedback was approximately
`(-1.8444, -0.7124) m`, an error of about **2.26 cm**, inside the configured 3 cm
tolerance. `/tidybot/cmd_vel` returned to zero publishers after shutdown and the Isaac
receiver remained present.

The first live attempt exposed that the scene advanced at roughly one third real time,
so the old 0.15 m/s command and 30 s wall timeout stopped at about 1.11 m. The main
simulation profile now uses a 0.30 m/s cap and 60 s limit. The completed retry exposed
a late cloud timestamp regression after the pose had already entered tolerance; pose
feedback now marks `DONE` immediately, and a late cloud cannot replace that terminal
success with `FAULT`. The controller regression suite is now **25 passed**.
