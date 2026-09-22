# Social Safety Decision — OpenArm／Isaac Sim 交接

最後整理：**2026-09-22**。本週目標：將模擬環境完整切換到 Isaac Sim，讓假人把手伸到 OpenArm 前方，由既有 Safety Decision 觸發機械手臂迴避，並留下可重現的測試證據。

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
| Isaac LAN 三路 RGB／前相機估計點雲 | 9/17 曾量測約 4.95 Hz、每筆 12,000 點；[歷史結果](results/isaac_lan_readiness_2026-09-17.json) | 補 aligned metric depth、相機外參與手部障礙管線 |
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

## Isaac LAN：目前可用的影像／點雲入口

遠端 Isaac 必須播放並發布 `/camera/front/color/image_raw/compressed`。首次使用須先依交接清單下載 Video Depth 模型。

```bash
# repository 根目錄；此預設入口不建立 /joint_command publisher
bash scripts/run_isaac_gui.sh
```

開啟 <http://localhost:8080/>，其他電腦用本機當前 LAN IP。
此模式使用 **domain 0／Fast DDS**，是 RGB 估計深度、未校正尺度的點雲；沒有啟用手部偵測、MoveIt 或 Safety Decision 閉環。
請勿加入 `--isaac-joint-control` 進行例行交接測試；非預期移動仍待釐清。
完整連線診斷與歷史限制見 [Isaac LAN 說明](docs/isaac_lan_gui.md)。

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
