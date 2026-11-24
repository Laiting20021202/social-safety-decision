# Isaac Sim／OpenArm 接續工作清單

> 這是 2026-09-22 的交接快照。現行 ROS 2 目錄及 Docker 啟動方式見[首頁](../README.md)。

更新：2026-09-22。目的：接手者使用 David 的電腦，完成「Isaac 假人伸手 → RGB-D 感知 → Safety Decision → OpenArm 迴避」演示。**以下是待完成的整合與驗收計畫，不是已通過的 Isaac 操作流程。** 已通過範圍見 [本次驗證](../results/repo_handoff_2026-09-22.md)。

## 先分清楚兩個 Isaac 入口

| 入口 | 程式／場景來源 | 目前界面 | 用途 |
|---|---|---|---|
| LAN 外部 Isaac | `scripts/run_isaac_gui.sh` 接收另一台機器的場景；該遠端 USD／Action Graph 未包含於此 repo | `/camera/{front,left,right}/...`、`/joint_command` JointState | 9/17 曾驗證 RGB、估計點雲及小步關節移動；控制異常後停用 |
| repo 原生 Isaac | `openarm_isaac_sim_test/scripts/run_sim.py` 由 Python＋YAML 生成 | `/rgbd/*`、`/clock`、TF、trajectory actions | 有原始碼與歷史場景 smoke；須重新接上目前感知／動態避障 |

建議以 repo 原生場景作為可重現基線，再帶入展示用假人。若沿用 LAN 場景，先取得並版本化其場景生成來源、Action Graph 設定與資產取得方式；目前 repo 無法重建那個遠端場景。
兩者不能只改 topic 名稱就視為等價：LAN `/joint_command` 不是 MoveIt 的 `FollowJointTrajectory` action server。

## 0. 取得工作環境與基線

現有主機的 repo 位於：

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
git status --short --branch
git pull --ff-only origin main
bash scripts/check_repo.sh
source /opt/ros/humble/setup.bash
.venv/bin/python openarm_isaac_sim_test/scripts/validate_environment.py --json
ros2 pkg prefix moveit_ros_move_group
ros2 pkg prefix moveit_simple_controller_manager
```

若 `git pull --ff-only` 因本機修改或分歧失敗，先保留／提交工作再整合；不要 reset 或 force-push。
建議接續開 `git switch -c work/isaac-handoff`，完成一個驗收關卡就提交程式、設定與結果摘要。

9/22 本機確認 Python 3.10.12、ROS Humble、RTX 4060 Ti 16 GB、Isaac `4.5.0-rc.36+release.19112.f59b3005.gl` 位於 `/home/david/isaacsim`。
`moveit_ros_move_group` 存在，但系統 prefix 找不到 `moveit_simple_controller_manager`；舊 demo 曾使用局部解壓的套件，不能假設乾淨 terminal 也能找到。請先補齊相容套件：

```bash
sudo apt install ros-humble-moveit-simple-controller-manager
```

新電腦先按 [主 README](../README.md) 安裝 Python 套件；模型與 ROS／資產準備如下，從根目錄執行：

```bash
# LAN RGB 估計點雲需要；正式 RGB-D 不依賴學習式深度
bash scripts/download_models.sh --video-depth
# 正式手部感知需要
bash scripts/setup_edgetam.sh
bash scripts/download_edgetam_checkpoint.sh
bash scripts/setup_openarm.sh
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select realtime_3d_safety_decision
source install/setup.bash

cd openarm_isaac_sim_test
python3 -m pip install --user -e '.[dev]'
mkdir -p ros2_ws/src
# 只在依賴尚未匯入時執行；保留既有 checkout 的本機修改
vcs import ros2_ws/src < dependencies.repos
colcon build --base-paths ros2_ws/src --symlink-install
source install/setup.bash
export OPENARM_SIM_ROOT="$PWD"
export OPENARM_DESCRIPTION_ROOT="$PWD/ros2_ws/src/external/openarm_description"
export ISAAC_SIM_ROOT=/home/david/isaacsim
python3 scripts/fetch_libhand_asset.py
python3 scripts/prepare_assets.py
```

本次未重新下載模型、import 第三方依賴或執行完整 colcon build；這些是 runtime 準備步驟。
若使用其他電腦，修改 `ISAAC_SIM_ROOT`。官方 OpenArm revision 由 `dependencies.repos` 固定，請沿用，不同版本須另記錄差異。新 terminal 必須重新 source ROS 與兩個 workspace。

## 1. 先釐清控制異常（P0）

9/17 曾有小步移動／復位的成功紀錄，**之後回報非預期移動**；當天停掉 GUI／控制程序並移除預設 `--isaac-joint-control`。原因尚未確認，成功紀錄不能覆蓋後續異常。

接續調查項目：

- 記錄 simulator host、版本、scene／graph、controller；確認沒有實體控制器接在同一 ROS graph。
- 在不發命令時檢查 `/joint_command` publishers／subscribers、`/joint_states` 來源及重複節點。GUI 中的 subscriber 名稱檢查只是一項程式條件，不能證明端點就是指定機器。
- 核對 14 個 arm joint 名稱、順序、rad 單位、URDF limits、Articulation Controller 的 name/index mapping；確認只傳一個 joint 的訊息是否會改變其他關節目標。
- 檢查多個 controller 同時寫入、Reset／Play 舊目標、初始 drive target、重連與回饋逾時等情況；目前沒有證據指定其中一項為根因。
- 在隔離的模擬場景重現並修正後，保存完整 14 關節 before/after、指令與時間戳，再解鎖受控移動測試。

LAN 只讀診斷（從根目錄）：

```bash
source scripts/isaac_ros_env.sh
ros2 topic list -t --no-daemon --spin-time 10
ros2 topic info /joint_command --verbose
ros2 topic info /joint_states --verbose
timeout 10 ros2 topic echo /joint_states --once --no-daemon
timeout 10 ros2 topic echo /clock --once --no-daemon
```

`isaac_ros_env.sh` 固定 domain 0／Fast DDS，依目前路由產生本機 DDS profile。`ros2_lan_env.sh` 固定 domain 42／Cyclone DDS；`run_koch_stream.sh` 也會 source 它，僅設定 `ROS_DOMAIN_ID=0` 仍會被覆寫。**需新增／修改 Isaac RGB-D 啟動器，統一整條管線的 ROS domain／RMW；這是未完成項目。**

## 2. Isaac 場景、假人與相機（P1）

先在沒有 ROS 的本地場景驗證載入與 PhysX 穩定性。在已設好上述環境的 `openarm_isaac_sim_test/` 下：

```bash
"$ISAAC_SIM_ROOT/python.sh" scripts/create_scene.py --headless --warmup-frames 240
"$ISAAC_SIM_ROOT/python.sh" scripts/run_sim.py --headless --no-ros \
  --scenario no_obstacle --max-steps 240 --report-contacts
```

這兩條命令會啟動本地模擬、生成／載入資產，但不連 ROS 控制；本次整理未執行。通過條件：沒有缺失 mesh/reference、關節爆開或意外接觸，base／table／camera pose 與設定一致。

接著完成：

- 原始碼入口：`openarm_sim/isaac_scene.py`、`scripts/run_sim.py`、`ros2_ws/src/openarm_isaac_bridge/openarm_isaac_bridge/runtime.py`。
- `config/scene.yaml`、`camera.yaml`、`hand_scenarios.yaml` 含沿用 Gazebo 的位姿；需在 Isaac 重新驗證，不能直接宣稱相機已校正。
- 現有 LibHand 是手部資產，不是完整假人。補假人與伸手動畫／可控關節，記錄資產授權、下載／重建方式。先用 LibHand 排查管線可行，但最終展示要符合「假人伸手」。
- 先測 `no_obstacle`，再測手部在相機內可見的姿勢與慢速伸入。`right_side_sweep` 等 scenario 可作基底；`PRE_GRASP`／`TRANSIT` 觸發情境需配合 task state，否則不會按預期開始。

通過條件：RGB、aligned depth、CameraInfo 同步；depth 為 `32FC1` 公尺；`world → robot links`、`world → rgbd_color_optical_frame` 可查；關節回饋持續更新；模擬暫停／reset 後時間處理正常。

## 3. 感知與 Safety Decision 的接線（P1）

以下為 repo 原生 Isaac 的目標契約；LAN 場景須先提供相同資料或明確的 adapter。

| 輸入／輸出 | 消費者／驗證重點 |
|---|---|
| `/rgbd/color/image_raw`、`/rgbd/aligned_depth_to_color/image_raw`、`/rgbd/color/camera_info` | 感知使用同一測量時間、有效 K、對齊尺寸；不要用 RGB 估計點雲假裝 metric depth |
| `/realtime_safety/pointcloud`、`/realtime_safety/environment_cloud_world` | 由目前 RGB-D 重建；光學座標 x右/y下/z前與 world transform 需一致 |
| `/edgetam_tracker/obstacle_cloud` | EdgeTAM／手部語意門檻後的量測障礙；檢查 self-filter 不會去掉真手 |
| `/realtime_safety/yolo_obstacles/pointcloud` | 歷史名稱的 mux 輸出，實際可選 EdgeTAM；應只有一個 publisher |
| `/perception/obstacles` | `rgbd_mask_adapter` 輸出至 safety bridge／動態規劃；明確設定上游 cloud，避免多條路同時發布 |
| `/openarm/safety/state`、`/openarm/dynamic_avoidance/status`、`/openarm/events` | 保留與障礙、指令、關節回饋同時間軸的診斷證據 |
| `/sim/ground_truth/*` | perception 驗收時僅 evaluator 可使用，不提供給 safety／planner |

需修改的具體缺口：

1. `configs/isaac_lan.yaml` 為 `mode: reconstruction`，`obstacle_perception.enabled: false`、`openarm.enabled: false`。這個 viewer 不啟動避障閉環。新增 Isaac RGB-D profile／啟動器，參考 `scripts/run_koch_stream.sh` 的 RGB-D 參數，處理 domain 與模擬時間。
2. `launch/perception_validation.launch.py` **不會啟動本 repo 的 EdgeTAM／GUI 感知程序**，須另整合。其 adapter 預設讀 `/edgetam_tracker/obstacle_cloud`，timeout 0.25 秒；Gazebo 路徑則用 realtime resampler 及 3.2 秒。依量測延遲選擇並驗證時效，不要為了通過測試任意延長。
3. Gazebo 的 `realtime_obstacle_resampler` 在 `launch/gazebo_phase1.launch.py` 配置，Isaac launch 沒有同等整合。移植前核對輸入是否仍為真實 RGB-D／模型輸出。
4. OpenArm 迴避使用子專案 `openarm_safety_bridge`＋`openarm_dynamic_avoidance`＋MoveIt task。根目錄 `pipeline/safety_decision.py` 的路徑決策與 OpenArm 關節控制不是同一個輸出接口。

診斷命令（所有節點已啟動、同一 ROS 環境後）：

```bash
ros2 run tf2_ros tf2_echo world rgbd_color_optical_frame
ros2 node info /openarm_safety_bridge
ros2 topic info /perception/obstacles --verbose
python3 scripts/validate_rgbd_realtime.py --duration 10 --require-obstacle
```

最後一條從子專案執行；此 validator 沿用既有 topic 契約，只有在 Isaac 提供所需 topics 後才適用。驗收要證明：手入鏡時產生對應非空障礙，移走／遮擋／斷流時依明訂規則處理，時間戳新鮮，沒有讀 GT 當模型結果。

## 4. 接上 MoveIt 與迴避（P1）

既有控制 action：

```text
/left_joint_trajectory_controller/follow_joint_trajectory
/right_joint_trajectory_controller/follow_joint_trajectory
/left_gripper_controller/gripper_cmd
/right_gripper_controller/gripper_cmd
```

原生 Isaac `IsaacRosBridge` 有 action 實作；先查 `ros2 action list -t`，確認回饋與取消語意。LAN JointState jog 必須另接 trajectory bridge，不可當規劃器替代品。

`perception_validation.launch.py` 目前啟動舊 `sorting_task`，預設 `auto_start:=true`；Gazebo 的保持／迴避／回復邏輯則在 `openarm_sorting_task/pose_goal.py`，由 `gazebo_phase1.launch.py` 啟動。**需為 Isaac 整合 `pose_goal` 與目標／planning scene 更新，不能只啟動舊 sorting launch 就宣稱與 Gazebo 同等能力。**

既有 launch 僅供接線除錯；完成前述關卡並確定隔離模擬後，才執行下列候選命令（不自動開始 sorting，但會啟動 simulator／controller／安全節點）：

```bash
# openarm_isaac_sim_test/；ROS workspace 與 Isaac 環境已 source/export
ros2 launch openarm_sim_bringup perception_validation.launch.py \
  scenario:=no_obstacle headless:=false allow_hsv_placeholder:=false auto_start:=false
```

正常迴避須有可觀察的 collision-checked 運動，而非僅顯示 WARNING／STOP。先驗單臂，再雙臂，確認另一臂保持原姿態；手撤離後回復保存目標。完全阻擋或 perception／joint state 過期時應停止，不能用 GT 靜默補值。門檻與逾時參數來源是 `config/safety_zones.yaml`，設定快照須隨驗收保存。

## 5. Isaac 專用驗收與交付（P2）

`scripts/run_dynamic_avoidance_test.sh`、`validate_dual_target_motion.py` 使用 Gazebo model／目標與 hand sweep 控制，**尚不能直接在 Isaac 使用**。`run_sim.py` 的手動 speed 指令也只接受 0.1／0.3／0.6，Gazebo wrapper 的 `speed:0.01`／`auto_sweep` 不能直接照搬；低速路徑可先用 YAML scenario 的 `linear_speed`。
請將 evaluator／hand control／target pose 抽出或新增 Isaac runner。`run_validation_suite.py --dry-run` 只列命令；現有非 dry-run 以 metrics 檔存在判斷完成，仍須檢查內容是否達標，不能只看 exit code。

本週最低驗收：

- [ ] Isaac 中包含 OpenArm、假人與可重現的伸手動作，沒有 Gazebo 程序依賴。
- [ ] 無手時可到達並保持目標；手伸入前方時，模型障礙輸出與影像位置一致。
- [ ] Safety Decision 取消／重規劃後，OpenArm 實際迴避；記錄最小間距、觸發到運動延遲與規劃失敗次數。
- [ ] 手撤離後恢復保存目標；另驗完全阻擋、depth／障礙斷流、joint state 過期等停止行為。
- [ ] 正常伸手／撤手情境至少連續三次成功，GT clearance > 0，無接觸、非預期關節動作或非預期 E-stop。故障注入的停止另列結果。
- [ ] 錄影同時看得到假人、OpenArm、感知障礙與 safety state；rosbag 保存 RGB-D、CameraInfo、TF、clock、joint states、障礙、安全狀態及指令／回饋；GT 僅供 evaluator。
- [ ] 每次記錄 commit ID、Isaac／模型版本、scenario seed／設定、結果 JSON、失敗原因及重現命令。

大型影片／bags 放 `openarm_isaac_sim_test/results/<run-id>/`（Git 忽略），另存可存取的交接位置；將小型摘要與檔案連結寫到根目錄 `results/` 並提交。更新主 README 的狀態表及本清單，不要把新量測寫成歷史紀錄。

## 排查入口

| 症狀 | 先查 |
|---|---|
| 沒有 topic／只有部分節點 | domain 0 vs 42、Fast DDS vs Cyclone、LAN route、Isaac 是否 Play；節點名存在不等於正在出資料 |
| GUI 有點雲但手臂不避障 | 是否仍在 `isaac_lan` reconstruction-only；模型 cloud → adapter → safety → planner → action 是否接通 |
| 一啟動就停止 | 障礙與關節回饋的 stamp、`use_sim_time`、TF、timeout；先修資料來源 |
| MoveIt 有計畫但不移動 | controller manager plugin、action 名稱／型別、joint 名稱、feedback／cancel |
| 看得到手但沒障礙 | RGB 可見性、MediaPipe gate、EdgeTAM checkpoint、metric depth、self-filter；COCO YOLO 沒有 hand class |
| 非預期移動再發生 | 停止該控制程序並暫停 Isaac，保留 logs／graph／命令與關節回饋，回到 P0 |
