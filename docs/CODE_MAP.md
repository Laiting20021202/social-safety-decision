# 程式索引

以 Docker 安裝後的 ROS 2 入口為起點，先查[系統架構](ARCHITECTURE.md)和[執行方式](DOCKER.md)。下表使用專案內的模組名稱，路徑相對於工作樹。

## 通用感知與主機應用

| 路徑 | 責任與主要入口 |
|---|---|
| `app.py` | 解析相機、模型、ROS 和 GUI 選項，建立 `RealtimePipeline` 及 ROS adapter。歷史 viewer 與邊緣電腦啟動腳本由此進入。 |
| `realtime_safety/config.py`、`types.py`、`performance.py`、`scheduler.py` | 設定／資料類別、計時與資源統計，以及把取像、推論、融合和本機安全輸出串起來的排程。 |
| `realtime_safety/pipeline/video_source.py`、`frame_queue.py`、`camera_motion.py` | 讀影像、管理影格佇列、估計相機運動。 |
| `pipeline/segmentation.py`、`tracker_2d.py`、`tracker_3d.py`、`obstacle_3d.py` | 偵測與遮罩、2D／3D 時序追蹤，將觀測組成 3D 障礙。 |
| `pipeline/monocular_depth.py`、`video_depth.py`、`pointcloud.py`、`ground_plane.py` | 單目與時序深度、點雲投影、地面估計。 |
| `pipeline/apriltag_calibration.py`、`rgb_tag_depth_scale.py`、`robot_self_filter.py` | 標記校正、估計深度比例尺、自身幾何排除。 |
| `pipeline/mast3r_slam_adapter.py`、`mast3r_slam_worker.py`、`st4rtrack_adapter.py` | 重建與歷史軌跡的模型介面，管理輸入與推論程序。 |
| `pipeline/danger_zone.py`、`traversable_region.py`、`local_planner.py`、`safety_decision.py` | 應用層危險區、可通行區、局部路徑候選與安全建議。 |
| `pipeline/tidybot_retreat.py` | 底盤退避狀態與速度計算，由專用的 ROS 2 節點接到 Tidybot。 |
| `realtime_safety/edgetam_tracker/tracked_obstacle_node.py` | ROS 2 點雲追蹤主入口，組合去雜訊、幾何追蹤、手部語意與 EdgeTAM 細化。 |
| `edgetam_tracker/sensor_sync.py`、`ros_utils.py`、`models.py` | 校對資料時效與感測器時間，轉換 ROS 訊息與內部觀測型別。 |
| `edgetam_tracker/pointcloud_preprocessor.py`、`cluster_extractor.py`、`pointcloud_tracker.py`、`quality.py` | 背景與點雲清理、聚類、軌跡關聯與品質檢查。 |
| `edgetam_tracker/hand_semantic_gate.py`、`intrusion_gate.py`、`robot_self_filter.py`、`urdf_self_filter.py` | 管理手部候選、侵入觸發與機器人移除。 |
| `edgetam_tracker/edgetam_wrapper.py`、`mask_pointcloud_fusion.py`、`projection_utils.py` | 官方 EdgeTAM 模型載入、跨影格執行，以及在影像與點雲間投影和驗證遮罩。 |
| `realtime_safety/ros2_bridge/runtime.py`、`safety_bridge_node.py`、`stamps.py` | 建立應用 ROS 執行環境，轉接安全資料與時間戳。 |
| `ros2_bridge/image_publisher.py`、`image_subscriber.py`、`pointcloud_publisher.py`、`rgbd_scene_bridge.py` | 接收及發布相機資料、重建和轉換 RGB-D 場景。 |
| `ros2_bridge/obstacle_cloud_mux.py`、`edgetam_control.py`、`edge_relationship.py`、`relationship_publisher.py` | 選用障礙資料來源，控制模型狀態與發布物體空間關係。 |
| `ros2_bridge/openarm_control.py`、`openarm_joint_state.py`、`robot_geometry.py`、`simulator_pose_bridge.py`、`tidybot_retreat.py` | 舊 viewer 控制與回饋、機器人座標及幾何，還有底盤退避 ROS 節點。 |
| `realtime_safety/gui/` | `dashboard` 與 `gui_state` 組裝 Viser 畫面，`scene_3d`、`reconstruction_scene`、`metric_bev`、`video_overlay` 繪製場景與診斷，`openarm_scene` 和 `isaac_joint_control` 供主機操作。 |
| `realtime_safety/export/`、`utils/` | 輸出點雲、原始結果及影片，共用幾何、GPU、計時與驗證工具。 |

## 模擬、ROS 2 規劃與控制

下列套件位於 `openarm_isaac_sim_test/ros2_ws/src/`。包內的 `setup.py` 或 `CMakeLists.txt` 列出可執行入口，`package.xml` 描述相依關係。

| 模組 | 程式責任 |
|---|---|
| `openarm_control_bringup/launch/control.launch.py` | 選擇 Isaac／硬體時鐘、量測來源、MoveIt 與任務入口。 |
| `openarm_sorting_task/node.py`、`grasp.py` | 既有分揀狀態機、抓取及落點邏輯。 |
| `openarm_sorting_task/pose_goal.py`、`route_planner.py`、`idle_avoidance.py` | 目標與重新規劃，避讓路線和待命姿態恢復。 |
| `openarm_dynamic_avoidance/node.py`、`policy.py` | ROS 軌跡與障礙監控，計算閘控、調速、重新規劃。 |
| `openarm_safety_bridge/node.py`、`policy.py`、`geometry.py`、`visibility_memory.py` | 接收有效觀測，維護安全狀態、距離、障礙物軌跡與可見性記憶。 |
| `openarm_perception_adapter/node.py`、`realtime_obstacle_resampler.py` | 轉接 RGB-D 遮罩或點雲，以及歷史 Gazebo 整合的更新節奏。 |
| `openarm_isaac_bridge/runtime.py`、`messages.py`、`control.py` | 感測與時鐘發布、訊息序列化、左右手臂及夾爪 action。PhysX 操作由模擬執行緒執行。 |
| `openarm_sim_bringup/isaac_planning_scene.py` | 把 Isaac 工作台的配置發布成 MoveIt 碰撞幾何。 |
| `openarm_sim_bringup/gazebo_*.py` | 歷史 Gazebo 的假手、初始姿態、RGB-D、GUI gateway、場景同步與規劃場景。 |
| `openarm_sim_evaluator/node.py` | 記錄真值、時序、事件及可重現的評估輸出。 |
| `hand_obstacle_controller/node.py`、`keyboard.py` | ROS 2 模擬場景服務、鍵盤與控制命令。 |

| 模擬 Python 程式 | 職責 |
|---|---|
| `openarm_sim/config.py`、`contracts.py` | YAML 載入、模式及訂閱隔離規則。 |
| `openarm_sim/assets.py`、`scene_model.py`、`isaac_scene.py` | 解析 OpenArm 資產，依配置建立工作台、標記、物件、假手及感測器。 |
| `openarm_sim/camera_math.py` | 內參、光學座標、深度回推與相機投影。 |
| `openarm_sim/scenario.py`、`hand_runtime.py` | 可重現的障礙運動腳本、手動命令與狀態。 |
| `openarm_sim/state_machine.py`、`metrics.py` | 純 Python 狀態／事件模型及試驗資料儲存。 |
| `openarm_isaac_sim_test/scripts/run_sim.py` | 建立 SimulationApp、場景和橋接，推進物理步，連接場景與 I/O。 |

## 部署、測試與歷史入口

| 路徑 | 目的 |
|---|---|
| `Dockerfile`、`compose.yaml`、`docker/` | 容器與 DDS 設定、建置、啟動和 Isaac ROS 合約驗收。 |
| `launch/edgetam_pointcloud_tracker.launch.py`、`scripts/edgetam_pointcloud_tracker_node` | 啟動現有 ROS 2 追蹤節點，選擇 live 或 bag 資料。 |
| `scripts/setup*.sh`、`scripts/download*.sh` | 補宿主機套件、官方模型和 OpenArm 資產。 |
| `scripts/run_{fast,agx,viewer,isaac_gui,koch_stream,rqt_camera,rviz_pointcloud,tidybot_retreat}.sh` | 過往主機與平台入口。LAN 環境設定由 `*_ros_env.sh` 或 `ros2_lan_env.sh` 提供。 |
| `scripts/configure_koch_camera.py`、`install_koch_wifi_stability.sh` | Koch 相機與網路現場設定。 |
| `scripts/benchmark.py`、`evaluate_rgb_metric_feasibility.py`、`measure_pointcloud_topic.py`、`tools/` | 量測推論、比例尺、ROS topic 與手部追蹤成效。 |
| `openarm_isaac_sim_test/launch/`、`scripts/start_*demo.sh` 和 `stop_*demo.sh` | 舊的 Isaac/Gazebo 啟動、關閉與整合測試。新容器啟動走 `openarm_control_bringup`。 |
| `openarm_isaac_sim_test/scripts/create_*`、`prepare_*`、`fetch_libhand_asset.py`、`export_stage.py` | 生成模擬資產、取得舊 Gazebo 手部網格與匯出 USD。 |
| `openarm_isaac_sim_test/scripts/validate_*`、`capture_gazebo_layout.py`、`evaluate_gazebo_contacts.py`、`record_gazebo_trial.py`、`spawn_recording_camera.py` 和 `run_*validation* / run_*test.sh` | 場景驗收、接觸與控制紀錄、測試執行和錄影。各程式的 `--help` 及[歷史操作手冊](history/2026-10-15-host-workflows.md)列出適用後端。 |
| `tests/`、`openarm_isaac_sim_test/tests/`、`scripts/check_repo.sh` | 根專案感知／ROS／GUI 回歸、模擬與控制測試及離線檢查。 |
| `config/`、`configs/`、`openarm_isaac_sim_test/config/` | ROS 感知設定、主機應用設定和場景／機器人配置。 |
| `results/`、`benchmarks/` | 前期評估摘要與跑分。相關版本見[歷史交付索引](DELIVERABLES.md)。 |

`pipeline/`、`edgetam_tracker/`、`ros2_bridge/` 是 `realtime_safety/` 的子路徑。`openarm_sim/` 與模擬的驗收腳本位於 `openarm_isaac_sim_test/`。
