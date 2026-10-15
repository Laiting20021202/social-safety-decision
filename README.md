# Social Safety Decision — OpenArm／Isaac Sim 交接

最後整理：**2026-09-30**。本週目標：將模擬環境完整切換到 Isaac Sim，讓假人把手伸到 OpenArm 前方，由既有 Safety Decision 觸發機械手臂迴避，並留下可重現的測試證據。

**目前尚未完成 Isaac Sim 感知 → Safety Decision → OpenArm 迴避的端到端驗收。**
既有 Gazebo 整合、Isaac 場景／ROS bridge 原始碼、Isaac LAN 影像與估計點雲可以接續使用；看到點雲或手動關節移動，不等於避障閉環完成。

接手請先讀 [Isaac Sim 工作清單、啟動前置與驗收方式](docs/ISAAC_SIM_HANDOFF.md)。
本機路徑：`/home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision`。
GitHub：[Laiting20021202/social-safety-decision](https://github.com/Laiting20021202/social-safety-decision)，交接分支為 `main`。

## 目前進度與阻塞

| 項目 | 狀態／證據 | 接續工作 |
|---|---|---|
| RGB-D、EdgeTAM／MediaPipe、人手點雲與追蹤 | 已有程式與單元測試；Gazebo 歷史整合見 [紀錄](openarm_isaac_sim_test/docs/HANDOFF_STATUS.md) | 在 Isaac 的同步 RGB-D、TF 下重驗 |
| Safety Decision、MoveIt 動態迴避／回復 | 已有 `openarm_safety_bridge`、`openarm_dynamic_avoidance`、`openarm_sorting_task` | 接上 Isaac trajectory action，驗證取消、迴避、回復與逾時停止 |
| Isaac 原生場景與 ROS bridge | 原始碼存在；歷史場景 smoke 有紀錄，本次未重跑 runtime | 重建資產、核對現行場景、相機與控制介面 |
| Isaac LAN 三路 RGB／前相機估計點雲 | 9/17 曾量測約 4.95 Hz、每筆 12,000 點；9/30 已填入前相機對底盤外參；[歷史結果](results/isaac_lan_readiness_2026-09-17.json) | 補 aligned metric depth、校正 RGB 估計點雲尺度與手部障礙管線 |
| Isaac LAN 關節步進面板 | 保留實驗程式，**預設停用**；9/17 有非預期移動回報，原因未確認 | 先釐清控制來源／joint mapping／部分關節指令語意，再做隔離模擬測試 |
| 假人伸手 → 感知 → OpenArm 迴避 | **未驗收**；現有資產以 LibHand 手掌／碰撞代理為主，非完整假人 | 補完整假人或確認展示資產，移植人手測試並保存證據 |
| 本次離線測試 | 主程式 **310 passed, 1 skipped**；模擬／控制 **105 passed** | [重現指令與驗證範圍](results/repo_handoff_2026-09-22.md) |

9/22 檢查時 `isaac-lan-gui.service`、`realtime-safety-3d.service` 均為 inactive，8080 無 listener。
這是檢查當下的狀態，不代表遠端 Isaac 已開啟。本次整理沒有啟動控制或執行機械手臂移動。

## 接手後的第一步

使用現有電腦時，先保留本機尚未提交的修改，再取得 `main`：

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
git status --short --branch
git pull --ff-only origin main
bash scripts/check_repo.sh
source /opt/ros/humble/setup.bash
.venv/bin/python openarm_isaac_sim_test/scripts/validate_environment.py --json
```

`check_repo.sh` 僅執行原始碼／shell 檢查及兩套 pytest，不啟動 ROS 節點、GUI 或模擬器。
其中神經模型／CUDA smoke 需要另提供 `SAFETY_SMOKE_VIDEO`；一般離線測試會跳過。

全新電腦先準備 Ubuntu 22.04、ROS 2 Humble、Python 3.10 與專案原先使用的 Isaac Sim 4.5：

```bash
git clone --branch main https://github.com/Laiting20021202/social-safety-decision.git
cd social-safety-decision
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[test]' -e './openarm_isaac_sim_test[dev]'
bash scripts/check_repo.sh
```

模型、第三方 repo、USD、ROS build/install 與錄影／rosbag 不在 Git；模型與模擬環境的補齊指令見 [交接清單](docs/ISAAC_SIM_HANDOFF.md)。本機雖有 `moveit_ros_move_group`，系統 ROS prefix 尚找不到 `moveit_simple_controller_manager`，不可把單元測試通過當成 runtime 已備妥。

## 每次啟動：Isaac 相機＋點雲 GUI

適用於 9/23 已實測成功的現有電腦環境；新電腦須先完成前述安裝與模型下載。
這是 Safety Decision 專案的 **影像／點雲 viewer**，完整機械手臂避障的狀態另見下一節。

### 1. Isaac 電腦：開場景並按 Play

兩台電腦接同一 LAN，Isaac 的 ROS domain 設為 `0`。目前場景的實際介面是：

| 用途 | Topic | 型別 |
|---|---|---|
| 前方相機 | `/camera/front/color/image_raw` | `sensor_msgs/msg/Image` |
| 前方相機參數 | `/camera/front/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| OpenArm／底盤關節回饋 | `/tidybot/joint_states` | `sensor_msgs/msg/JointState` |
| 模擬時間 | `/clock` | `rosgraph_msgs/msg/Clock` |
| 本機重建點雲（GUI 啟動後） | `/realtime_safety/pointcloud` | `sensor_msgs/msg/PointCloud2` |

目前沒有 compressed 相機 publisher；GUI 預設接 raw。前方相機須隨 Isaac Play 持續發布。

### 2. GUI 電腦：重開機後套用影像接收設定

在 David 這台電腦的終端執行一次；**不是在 Isaac 電腦執行**。
目前同一次開機若已套用 `3`／`134217728`，可略過 sudo 指令：

```bash
sysctl net.ipv4.ipfrag_time net.ipv4.ipfrag_high_thresh
sudo sysctl -w net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

密碼輸入在終端。這是整台接收端的暫時 IPv4 分片設定，解決本次 Wi-Fi raw 影像
長時間收不完整的問題；未寫入永久設定。回復原值與細節見 [LAN 說明](docs/isaac_lan_gui.md)。

### 3. 終端 A：啟動 GUI，保持終端運行

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
bash scripts/run_isaac_gui.sh
```

腳本會自行設定 Humble、domain 0／Fast DDS，並使用專案 `.venv` 啟動相機接收、
GPU 深度估計、點雲生成與網頁服務。瀏覽器開啟 <http://localhost:8080/>，
在 **Camera Input** 看 RGB 畫面，主視窗看彩色點雲。

另一個終端也可直接開瀏覽器：

```bash
xdg-open http://localhost:8080/
```

其他電腦用 `http://<GUI電腦當前LAN-IP>:8080/`；本機執行 `hostname -I` 可查 IP。
若 GUI 已運行，直接開網頁即可，不要再啟動第二份程式佔用 8080。

### 4. 終端 B：訂閱關節（選用，不影響 GUI 影像）

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
source scripts/isaac_ros_env.sh
ros2 topic echo /tidybot/joint_states sensor_msgs/msg/JointState \
  --qos-reliability best_effort --qos-durability volatile --no-daemon
```

`name[i]` 對應 `position[i]`，旋轉關節使用 rad。這個命令只讀回饋，不發送控制。
它也不會把關節資料送進 GUI；目前 `isaac_lan` 關閉 OpenArm 模型疊圖與控制。

### 5. 終端 C：確認相機與點雲更新（選用）

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
source scripts/isaac_ros_env.sh
ros2 topic list -t --no-daemon --spin-time 10

# 檢查一張完整相機訊息，省略巨大的影像資料陣列
timeout 15 ros2 topic echo /camera/front/color/image_raw sensor_msgs/msg/Image \
  --qos-reliability best_effort --qos-durability volatile \
  --no-daemon --once --no-arr

# 持續監看點雲接收率，Ctrl+C 結束
ros2 topic hz /realtime_safety/pointcloud --wall-time
```

9/23 修復後量到影像約 4.69 Hz、點雲約 4.89 Hz，每筆 12,000 點；這是當時的
量測，不是保證 FPS。看到 topic 名稱不等於收到完整影像。若再次卡住，依
[LAN 排查步驟](docs/isaac_lan_gui.md) 檢查；證據見 [修復紀錄](results/isaac_image_receive_2026-09-23.md)。

所有 Isaac 監看終端都 source `isaac_ros_env.sh`。不要混用 `ros2_lan_env.sh`／
`run_koch_stream.sh`，它們會切換到舊 domain 42／Cyclone DDS。

### 停止與下次重啟

在終端 A 按 `Ctrl+C` 停止 GUI，在 B／C 按 `Ctrl+C` 停止各自的監看。
關閉瀏覽器不會停止後端。下次仍在同一次開機，只需確認 Isaac Play 並重做步驟 3；
重開機後先重做步驟 2。若之後重新啟用 systemd 背景 GUI，手動啟動前需先停止該服務。

## Tidybot 底盤：1 公尺內障礙觸發反向 2 公尺退避（9/30）

新增獨立底盤退避節點，與 OpenArm 關節避障分開。先啟動上面的 GUI，再開一個終端：

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision
bash scripts/run_tidybot_retreat.sh
```

**目前預設為 dry-run，不發送 `/tidybot/cmd_vel`**；讀取點雲與
`/tidybot/joint_states`，在 `/realtime_safety/tidybot_retreat/status` 顯示擬發送速度。
Isaac 的相機外參、body-frame Twist、位置回饋、公尺單位與 0.5 秒接收端 watchdog 已確認。
實際移動前仍須校正 **RGB 單目估計點雲的公尺尺度**，並在當下場景確認整條 2 m 退路；
這兩項未標記驗證時 `--execute` 會拒絕啟動。

完成確認後，`--execute` 才會發布 Twist：連續 3 張點雲確認 1 m 內障礙，
以位置回饋朝觸發方向的反方向移動 2 m（目標容差 3 cm），到達／異常時送零速。
完成後需明確 reset，不會持續反覆後退。
9/30 已用獨立測試點雲完成左右兩向的真實 Isaac 閉環：兩趟均進入 `DONE`，
終點分別距 2 m 目標 2.66 cm 與 2.96 cm；[驗證紀錄](results/tidybot_retreat_2026-09-30.md)。
這證明底盤命令與位置回饋，但不代表正式 RGB 估計點雲的尺度已驗證。
完整設定、停止／reset 與驗證說明見 [Tidybot 退避操作](docs/TIDYBOT_RETREAT.md)。

## Safety Decision：OpenArm 完整避障要另外完成哪些步驟？

**上述 GUI 指令沒有啟用手部偵測、安全決策或 OpenArm 自動迴避。**
`configs/isaac_lan.yaml` 是 `mode: reconstruction`，且 `obstacle_perception.enabled`
與 `openarm.enabled` 均為 `false`。目前點雲為 RGB 估計深度；退避設定已有相機外參，
但尺度尚未驗證，OpenArm 避障鏈也尚未完成相同座標整合。
單純加入 `--mode safety` 或 `--isaac-joint-control` 不會補齊 Isaac 閉環；
後者仍有先前非預期移動問題待釐清。

Isaac 完整避障須依序完成以下整合，**目前還沒有已驗收的一鍵啟動命令**：

1. Isaac 發布同步 metric depth＋RGB＋CameraInfo，核對相機到機器人 TF 與尺度。
2. 啟動手部感知／EdgeTAM 與 adapter，持續輸出 `/perception/obstacles`。
3. 整合 `openarm_safety_bridge`、`openarm_dynamic_avoidance`、MoveIt 與 `pose_goal`。
4. 將現場 `/tidybot/joint_states` 接到控制鏈所需的關節介面，接通 Isaac trajectory
   action；`/tidybot/joint_command` 的 JointState 不是 `FollowJointTrajectory` action。
5. 驗證伸手時實際迴避、撤手後回復，以及輸入過期時停止。

各步的程式位置、launch 缺口與驗收方式見 [Isaac 交接清單](docs/ISAAC_SIM_HANDOFF.md)。

若要回歸**先前的 Gazebo 完整避障版本**，先完成子專案 README 的資產、模型、
ROS workspace、MoveIt controller plugin 與 `realtime-safety-3d.service` 環境準備。
停止上述 Isaac GUI 後，在獨立終端執行以下歷史入口；此流程會啟動模擬控制及移動測試，
不是把既有 LAN Isaac 場景接入 Safety Decision：

```bash
cd /home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision/openarm_isaac_sim_test
OPENARM_GUI_URL=http://localhost:8080/ bash scripts/start_gazebo_demo.sh
# 必須等到 OPENARM GAZEBO DEMO READY，才執行下列測試
bash scripts/run_dynamic_avoidance_test.sh
# 測試結束、需要停止 demo 時
bash scripts/stop_gazebo_demo.sh
```

以上 Gazebo runtime 本次未重跑；前置與操作見 [模擬專案 README](openarm_isaac_sim_test/README.md)。

## 本週工作順序與完成定義

1. 釐清 9/17 控制異常，確定只連到指定模擬器，記錄 Isaac 版本、場景、controller 與關節對應。
2. 在 Isaac 完成 OpenArm、假人伸手動作、RGB-D、CameraInfo、`/clock`、TF 與 `/joint_states`；確認尺度／座標。
3. 將真實 RGB-D 感知接到 `/perception/obstacles`，讓 Safety Decision／MoveIt 使用感知障礙；ground truth 僅供評估。
4. 接通 Isaac `FollowJointTrajectory`，逐項驗證目標保持、手接近時迴避、撤手後回復、輸入過期時停止。
5. 將 Gazebo 專用驗收改成 Isaac 可重現測試，至少連續三次通過並保存錄影、rosbag、JSON、設定與 commit ID，再更新本頁狀態。

各步的程式入口、已知缺口、命令與通過條件均在 [Isaac Sim 交接清單](docs/ISAAC_SIM_HANDOFF.md)。
現有 `run_dynamic_avoidance_test.sh` 是 **Gazebo 專用**，不能當成 Isaac 完成證明。

## Repository 導覽

| 路徑 | 用途 |
|---|---|
| `app.py`、`realtime_safety/` | 感知管線、GUI、ROS bridge；主程式內的 safety worker 與 OpenArm 控制層需區分 |
| `configs/isaac_lan.yaml`、`scripts/run_isaac_gui.sh` | Isaac LAN RGB／估計點雲 profile 與啟動器 |
| `config/edgetam_pointcloud_tracker.yaml` | ROS 手部／點雲追蹤參數 |
| `openarm_isaac_sim_test/openarm_sim/`、`config/` | Isaac 場景生成、手部路徑、相機／安全設定（`config/` 指該子專案內） |
| `openarm_isaac_sim_test/ros2_ws/src/` | Isaac bridge、動態避障、安全橋接、MoveIt task、evaluator |
| `tests/`、`openarm_isaac_sim_test/tests/` | 兩套獨立 pytest；請都執行 |
| `results/` | 可提交的小型驗證報告；大型 runtime 結果保留在被忽略的子專案 `results/` |

## 文件索引與歷史入口

- [Isaac Sim 交接清單](docs/ISAAC_SIM_HANDOFF.md)：目前工作優先順序與驗收。
- [分支分工與進度](docs/BRANCHES_AND_PROGRESS.md)：`main` 是本次交接快照；舊 layer 分支尚未同步本次變更。
- [模擬專案 README](openarm_isaac_sim_test/README.md)：Gazebo 回歸操作、既有 Isaac launch 與控制說明。
- [模擬歷史驗證](openarm_isaac_sim_test/docs/HANDOFF_STATUS.md)、[本次離線驗證](results/repo_handoff_2026-09-22.md)。
- [Viewer／Koch／模型技術手冊](docs/VIEWER_REFERENCE.md)：原 README 的完整技術內容與歷史效能資料。
- [EdgeTAM 接口與限制](docs/edgetam_pointcloud_tracker.md)、[OpenArm GUI 座標契約](docs/openarm_gui.md)。

歷史 Gazebo 回歸使用 `openarm_isaac_sim_test/scripts/start_gazebo_demo.sh`；此腳本會管理既有 demo 程序。它不是本週 Isaac 切換入口。舊 `2d_version`／`3d_version` 分支保留供比較。
