# OpenArm dual-target and dynamic-hand safety validation

This project is a deterministic OpenArm safety test harness with both Isaac Sim sources and
a lighter, currently active Gazebo Classic + RViz runtime. The current demo moves both
gripper TCPs to two gravity-free target cubes edited directly in Gazebo; cube picking is
intentionally paused. It is a
validation environment, not a training environment, and it does not run RL, PPO, imitation
learning, MπNets training, or dataset generation.

The source of truth is Python plus YAML. `worlds/openarm_sorting.usd` and
`assets/openarm_cache/` are generated caches and are intentionally ignored by Git.

> Safety scope: these tests measure behavior in simulation. Passing them is not a robot
> safety certification and is not a substitute for a risk assessment, validated safety PLC,
> guards, torque limits, or hardware E-stop.

## 重開機後快速啟動（Gazebo + 動態避障）

請在 Ubuntu 圖形桌面的 Terminal 執行。啟動腳本會自動停止舊程序、載入 ROS 2、
編譯有修改的套件、啟動 Gazebo、RViz、MoveIt、OpenArm controller、RGB-D 感知、
動態避障與 8080 GUI，不需要另外開多個 Terminal：

```bash
git clone --branch main https://github.com/Laiting20021202/social-safety-decision.git
cd social-safety-decision/openarm_isaac_sim_test
./scripts/start_gazebo_demo.sh
```

第一次啟動可能需要約 1–2 分鐘。必須等終端顯示以下訊息後再操作：

```text
OPENARM GAZEBO DEMO READY
```

然後開啟 <http://192.168.0.234:8080/>。若要直接執行自動化「移動人手侵入、
OpenArm 重新規劃並繞過障礙」測試，保持 Demo 運行並執行：

```bash
./scripts/run_dynamic_avoidance_test.sh
```

The perception safety bridge predicts the continuously tracked hand two
seconds ahead for planning, while immediate E-stop distance uses only the
current RGB-D cloud. Neural correction jumps are rate-limited to 0.20 m/s
with 0.02 m slack, so one-frame OpenArm false detections cannot teleport the
collision object onto the robot.

測試會自動使用 `Perception` 障礙來源：先命令雙臂抵達並持續停在各自目標，
再以 `0.01 m/s` 讓 Gazebo 人手在固定 X/Z 下沿 Y 軸左右等速循環。檢查 RGB-D 感知、
靠近時的逃逸/繞行、離開後回復原目標、目標誤差與 evaluator-only
ground-truth clearance，並將結果寫入 `results/dynamic_dual_target_<timestamp>.json`。測試進行時不要同時操作
Gazebo 或 GUI。

### GUI 手動操作

1. 在 Gazebo 使用 Translate 工具移動 `left_target_cube`、`right_target_cube`；兩顆
   方塊就是左右夾爪 TCP 的目標點。
2. 在 8080 GUI 的 **OpenArm Control** 選擇 Planner =
   **MoveIt + Dynamic Safety**、Obstacle Source = **Perception**。
3. 按 **MOVE LEFT TCP**、**MOVE RIGHT TCP** 或
   **MOVE BOTH TCPs TO GAZEBO CUBES**。
4. 將 **Hand max speed** 設為 **0.01 m/s**（目前動態規劃預設值），勾選
   **Auto left/right sweep (fixed height)**，即可讓人手以固定高度左右移動。
5. 正常行為是 TCP 抵達後留在目標；手接近時取消舊軌跡並執行向上/向外的
   collision-checked 逃逸；手離開後自動回到先前保存的目標關節姿態。只有按
   **HOME** 才會返回 Home。若真的進入緊急距離，系統會保持 `EMERGENCY_STOP`，先按
   **Withdraw**，再按 **RESET** 與 **HOME**。

目標方塊不會被程式偷偷投影或移動；其 Gazebo world pose 就是實際 TCP 目標。
若 GUI 顯示 `TARGET_UNREACHABLE`，代表 MoveIt 在官方關節限制內找不到 IK，而非
按鈕失效。此時應在 Gazebo 將方塊放回 OpenArm 工作空間，不能以放寬 joint limit
冒充成功。目前實測左臂對 `[-0.220, 0.180, 0.340]` 有 IK，而使用者目前的
`[0.104, 0.180, 0.138]` 在 600 個末端姿態樣本下皆無 IK。

### 停止、重新啟動與記錄

停止所有 Gazebo Demo 程序：

```bash
./scripts/stop_gazebo_demo.sh
```

要重新啟動可直接再次執行 `./scripts/start_gazebo_demo.sh`；它本身也會先清除舊的
Demo 程序。若重開機後 8080 感知服務沒有正常啟動，可執行：

```bash
systemctl --user restart realtime-safety-3d.service
./scripts/start_gazebo_demo.sh
```

故障時查看完整記錄：

```bash
tail -f .demo/logs/gazebo_demo.log
journalctl --user -u realtime-safety-3d.service -f
```

## What is implemented

- Official OpenArm v1.0 bimanual URDF import, pinned source revisions, gravity enabled, and
  PhysX self-collision enabled. Official SRDF adjacent pairs plus the v1.0 wrist enclosure
  pair (`link5`/`link7`) are filtered; all other non-adjacent pairs remain active.
- Programmatic table, yellow edge, detector-valid AprilTag, camera, lighting, a textured
  LibHand mesh, and hidden collision proxies. The active Gazebo task has exactly two
  gravity-free, collision-free TCP target cubes; the legacy Isaac sorting source remains.
- Generic 640×480 pinhole RGB-D topics, synchronized `32FC1` metre depth, CameraInfo,
  aligned depth, colored PointCloud2, `/clock`, TF, and configurable noise/drop/latency.
- Standard `FollowJointTrajectory` and `GripperCommand` action servers backed by the Isaac
  articulation, plus `/joint_states` feedback.
- MoveIt task states: `HOME → SELECT_OBJECT → PRE_GRASP → GRASP → LIFT → TRANSIT → PLACE
  → RETREAT → NEXT_OBJECT → DONE`, planning-scene attachment, cancellation, and recovery.
- Ten deterministic hand scenarios, UI buttons, ROS topic/services, and terminal keyboard.
- Strictly separate ground-truth and perception safety subscriptions.
- Evaluator JSON/CSV/config snapshots and an automated scenario/rosbag runner.

See [architecture](docs/architecture.md), [calibration](docs/calibration.md),
[validation protocol](docs/validation_protocol.md), and the exact
[environment audit](docs/environment_report.md).

## Current visible Gazebo demo

Fetch the licensed hand asset once, then launch everything from one terminal:

```bash
python3 scripts/fetch_libhand_asset.py
./scripts/start_openarm_demo.sh
```

This opens Gazebo and RViz and starts OpenArm controllers, MoveIt, RGB-D, the actual
MediaPipe + EdgeTAM perception process, the dynamic safety layer, and the existing web GUI
at `http://192.168.0.234:8080/`. Move `left_target_cube` and
`right_target_cube` with Gazebo's Translate tool, select `MoveIt + Dynamic Safety`, then
press `MOVE BOTH TCPs TO GAZEBO CUBES`. Each arm is planned against the other arm, the
table, and the selected live hand-obstacle source. The old pick command remains rejected
as `PICK_TASK_PAUSED`.

In dynamic mode a newly visible or displaced hand cancels the obsolete controller goal,
updates a padded MoveIt collision object, and requests a new collision-free spatial path.
If a moving hand reaches the soft-stop band during an active trajectory, the affected arm
first cancels the stale trajectory, then executes a dedicated MoveIt-checked TCP escape
that lifts at least 16 cm and shifts 10 cm away from the measured hand. Ordinary
trajectories remain blocked during `PAUSE`; only this upward/away escape can pass that
gate. The original marker/Home step remains queued and is replanned after clearance.
A finite three-second escape window prevents a conservative near-cloud point from
latching the robot at the pose being approached; if no checked escape makes progress,
the normal `EMERGENCY_STOP` still latches.
After the immediate escape it rebuilds a local side/upper/lower guide from the measured
TCP, limits backward target-direction motion to 2 cm, and keeps approaching the selected
marker.
Once a soft stop has selected this local side route, later moving-cloud
refreshes remain on the side route for the rest of that target request; they
cannot switch back to a slower under-hand descent.  Forced side routes use an
additional 6 cm lateral clearance.
All dynamically monitored paths are capped at 25% speed; the safety supervisor can reduce
that further. The monitor follows the active bypass waypoint rather than the obsolete
straight goal. A lower bypass uses approach, entry, exit, and clear-side recovery legs
before reconnecting to the precise target. Only the emergency distance remains a latched
hard stop. This is a MoveIt spatial-replanning layer, not a learned MπNets policy and not
a hardware safety certification.

Run the perception-only moving-intrusion acceptance test while the demo is active. This
wrapper loads the ROS environment and gives each run a timestamped result file:

```bash
./scripts/run_dynamic_avoidance_test.sh
```

The test passes only if both TCPs first touch and hold their targets, then the Gazebo hand
sweeps at 0.01 m/s with fixed X/Z, the RGB-D-derived obstacle causes collision-checked
escape/replanning, and both TCPs restore their saved targets after the hand leaves.
No automatic Home command is issued; evaluator-only ground-truth clearance must stay
positive and no emergency stop may occur.

Use Gazebo's Select/Translate/Rotate tools to move `rgbd_sensor`, `work_table`, the two
target cubes, `human_hand`, or the AprilTag directly. ROS TF, RGB-D world points, the web
viewer, and the MoveIt scene follow `/gazebo/model_states`. To preserve the current Gazebo
arrangement across restarts:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source ../scripts/ros2_lan_env.sh
./scripts/capture_gazebo_layout.py
```

OpenArm's fixed base is deliberately excluded from this capture because moving it without
regenerating the MoveIt robot model would break the planning-frame contract.

The production cloud path does not consume Gazebo's native `/rgbd/points` or
`/rgbd/points_world`. `realtime_3d_safety_decision` synchronizes RGB, aligned `32FC1`
depth, and CameraInfo, performs metric pinhole back-projection, and publishes the current
frame as `/realtime_safety/pointcloud` plus the camera-pose-transformed
`/realtime_safety/environment_cloud_world`. Hidden or occluded pixels therefore disappear
on the next depth frame instead of remaining in a simulator scene cache. Configurable
1.5 mm depth noise is enabled for physical-camera-like temporal variation; it is explicitly
simulation noise and is not evidence of sensor realism.

EdgeTAM independently receives the same RGB, aligned depth, and CameraInfo (its native
PointCloud2 input is empty). MediaPipe supplies the hand semantic gate and the OpenArm TF
self-filter removes the robot before clustering. The local YOLO checkpoints still exist,
but they are generic 80-class COCO object weights and contain no human-hand class; the GUI
now labels that distinction instead of presenting them as a hand detector. The confirmed
mask is refreshed against current depth for `/edgetam_tracker/obstacle_cloud_realtime`.
Verify stamps, rates, back-projection, current-frame occlusion, and model-confirmed hand
extraction with:

```bash
python3 scripts/validate_rgbd_realtime.py --duration 6
python3 scripts/validate_rgbd_realtime.py --duration 10 --require-obstacle
python3 scripts/validate_dual_target_motion.py
```

The one-click Gazebo demo moves the LibHand model into a known camera-visible
preview pose and requires a non-empty real-model obstacle cloud before it prints
`OPENARM GAZEBO DEMO READY`. Use **SHOW HAND + POINT CLOUD** to repeat this check
from the web GUI, or set `OPENARM_DEMO_HAND_PREVIEW=0` for an empty-scene startup.

The three target-motion buttons move each selected TCP to the current Gazebo marker
surface and keep it there. Returning Home is now explicit: press **HOME** when desired.
Always-on avoidance remains active while holding a marker. A nearby hand causes the
affected arm to keep TCP X/Y and tool orientation while executing a collision-checked
vertical lift; after the hand withdraws, the exact saved target joint posture is restored
automatically. This holding reflex does not insert a side route or Home leg. If a marker is outside the
official OpenArm joint workspace, finite planning retries end visibly instead of
silently leaving a request active.

## Host prerequisites

The audited host uses Ubuntu 22.04, ROS 2 Humble, Isaac Sim 4.5, Python 3.10, and an RTX
4060 Ti. Keep those working versions; do not upgrade Isaac Sim or Isaac Lab for this project.

The audited host uses the Humble binaries (`move_group`, OMPL, KDL and messages) from
`/opt/ros/humble`. Do **not** mix a MoveIt source overlay built against a different ABI
with the system installation; that makes the KDL plugin fail to load. Use one consistent
prefix:

```bash
source /opt/ros/humble/setup.bash
```

One matching MoveIt component is missing on the audited host. Install the exact Humble
package once (it is the controller-action adapter, not a hardware driver):

```bash
sudo apt install ros-humble-moveit-simple-controller-manager
```

## Setup

From this directory:

```bash
python3 -m pip install --user -e '.[dev]'
vcs import ros2_ws/src < dependencies.repos
source /opt/ros/humble/setup.bash
colcon build --base-paths ros2_ws/src --symlink-install
source install/setup.bash
export OPENARM_SIM_ROOT="$PWD"
export OPENARM_DESCRIPTION_ROOT="$PWD/ros2_ws/src/external/openarm_description"
```

If the official repositories already exist elsewhere, do not clone another copy; set
`OPENARM_DESCRIPTION_ROOT` to that checkout. `OPENARM_USD_PATH` can instead point to a
prepared official OpenArm USD. Run the read-only audit and asset resolver:

```bash
python3 scripts/validate_environment.py
python3 scripts/prepare_assets.py
```

The pinned dependencies are:

- `enactic/openarm_description` at `1fba2cbc05001f05b4514120b70130b4ac06f409`
- `enactic/openarm_ros2` at `4e837e1d0dae692ff67b560b69d8d281d7a8d4ed`

## Launch

Build, inspect, and export the stage without ROS:

```bash
export ISAAC_SIM_ROOT=/path/to/isaacsim
"$ISAAC_SIM_ROOT/python.sh" scripts/create_scene.py --headless --warmup-frames 240
"$ISAAC_SIM_ROOT/python.sh" scripts/run_sim.py --no-headless --no-ros \
  --scenario right_side_sweep
```

One-command ground-truth validation after the ROS workspace is sourced:

```bash
OPENARM_SIM_ROOT="$PWD" ros2 launch openarm_sim_bringup \
  ground_truth_validation.launch.py scenario:=right_side_sweep headless:=false
```

Headless mode without RViz:

```bash
OPENARM_SIM_ROOT="$PWD" ros2 launch openarm_sim_bringup \
  ground_truth_validation.launch.py scenario:=sudden_intrusion \
  headless:=true use_rviz:=false
```

Control the hand from another sourced terminal:

```bash
ros2 run hand_obstacle_controller hand_keyboard
ros2 service call /sim/hand/trigger_hand std_srvs/srv/Trigger '{}'
ros2 topic pub --once /sim/hand/command std_msgs/msg/String '{data: withdraw}'
```

## Perception-in-the-loop

The active Gazebo path uses real inference rather than simulator labels:

1. Gazebo publishes synchronized generic RGB, aligned metric depth, and CameraInfo.
2. `realtime_3d_safety_decision` regenerates current optical/world environment clouds from
   those images and runs MediaPipe hand confirmation plus the loaded EdgeTAM checkpoint.
3. Its measured dynamic-hand cloud is refreshed from the current depth image and relayed
   as `/perception/obstacles`; no simulator pose enters this perception path.
4. The safety bridge transforms that cloud into the world frame and updates the MoveIt
   Planning Scene; the dynamic layer cancels an obsolete path, requests a new spatial
   route, then rebases/retimes the new JointTrajectory from measured joints.

Start the formal mode:

```bash
OPENARM_SIM_ROOT="$PWD" ros2 launch openarm_sim_bringup \
  perception_validation.launch.py scenario:=right_side_sweep \
  allow_hsv_placeholder:=false
```

No safety node in perception mode subscribes to `/sim/ground_truth/*`. The optional HSV
adapter remains disabled and must not be reported as neural inference.

## Tests and validation suite

```bash
python3 -m pytest -q
python3 -m compileall -q openarm_sim scripts ros2_ws/src launch tests
python3 scripts/run_validation_suite.py --dry-run
python3 scripts/run_validation_suite.py --mode ground_truth
```

Fault injection is available without editing YAML:

```bash
OPENARM_CAMERA_LATENCY_MS=120 \
OPENARM_CAMERA_FRAME_DROP_PROBABILITY=0.25 \
ros2 launch openarm_sim_bringup perception_validation.launch.py \
  headless:=true use_rviz:=false
```

Each evaluator run writes:

```text
results/<UTC-run-id>/
├── metrics.json
├── trajectory.csv
├── events.csv
├── config_snapshot/
├── screenshots/
└── rosbag/
```

## Current verification boundary

| Capability | Status on audited host |
|---|---|
| Isaac stage, official OpenArm import, six deterministic cubes | Passed |
| 240-step PhysX stability, no contact pairs, no cube drift | Passed |
| Visible Gazebo + RViz, editable camera/table/item poses | Passed |
| 640×480 RGB, metre depth, aligned/world point clouds, 8080 live preview | Passed |
| MediaPipe hand confirmation + CUDA EdgeTAM measured obstacle cloud | Passed (8,376 points in the recorded probe) |
| Pure Python config/math/state/safety/evaluator tests | Passed |
| MoveIt world-XYZ TCP goal through Gazebo FollowJointTrajectory | Passed |
| Perception collision object + moving-obstacle spatial replanning | Passed (3 replans, then `HOME_REACHED`) |
| Repeated/preemptive HOME GUI command | Passed |
| Close, robot-occluded hand detection during intrusion | Not passed for every pose; visibility dependent |
| Cube pick/place and grasp reliability | Paused by design for the current phase |

Do not treat these simulation checks as hardware safety certification.

## Troubleshooting

- `OpenArm assets not found`: set `OPENARM_DESCRIPTION_ROOT` or `OPENARM_USD_PATH`, then
  rerun `scripts/prepare_assets.py`.
- `moveit_ros_move_group not found`: install the missing Humble component or rebuild a
  complete, ABI-consistent MoveIt workspace; do not mix it with `/opt/ros/humble` libraries.
- `move_action is unavailable`: MoveIt did not start; inspect the complete launch log before
  restarting the task node.
- Isaac ROS Python imports fail: start through `ros2 launch`, or source ROS before invoking
  `$ISAAC_SIM_ROOT/python.sh`.
- Perception immediately E-stops: this is the fail-safe timeout when no fresh
  `/perception/hand/points` arrives. Check RGB/depth/mask stamps and TF.
- OmniHub warnings while offline are non-fatal for local assets. Missing mesh/reference
  errors are fatal and must not be ignored.
