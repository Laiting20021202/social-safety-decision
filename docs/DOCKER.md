# Docker 與 Sim-to-Real

## 建置與驗收

在 Linux x86-64 的 NVIDIA GPU 主機安裝 Docker Compose 與 NVIDIA Container Toolkit。映像使用 Isaac Sim 4.5.0 與 ROS 2 Humble 的固定 digest。控制映像依 `openarm.repos` 取得 OpenArm 模型和 MoveIt 設定，以 `colcon build --base-paths src` 建置套件。初次建置從公開登錄檔與套件庫下載相依項目。[NVIDIA Container Toolkit 安裝說明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) · [上游套件版本](../openarm.repos)

```bash
bash docker/build.sh
ACCEPT_EULA=Y docker compose --profile isaac up -d
docker compose --profile isaac logs -f isaac
```

`ACCEPT_EULA=Y` 表示接受 [NVIDIA Omniverse 授權條款](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)。預設為 `PRIVACY_CONSENT=N`。Kit 的快取保存在 Compose volumes，第一次啟動會花較久的時間。

在另一個終端執行驗收。它會核對推進中的模擬時鐘、RGB-D 同步、光學 TF、關節回饋、感知、安全狀態以及 MoveIt、手臂和夾爪 action。程式會產生 JSON 並以非零狀態回報缺項。

```bash
mkdir -p outputs
set -o pipefail
docker compose exec -T control bash docker/entrypoint.sh \
  python3 docker/verify_isaac.py --timeout 150 | tee outputs/isaac-contract.json
```

預設無障礙場景可用 `--expect-clear` 檢查是否回到 SAFE 且點雲維持空。若要測試可見障礙，啟動時另設 `OPENARM_ISAAC_SCENARIO=perception_preview`，驗收程式加上 `--require-obstacle`。示範手部的辨識標記由 RGB 與對齊深度量測。學習模型可接在同一介面，再用其量測資料評估。使用外部感知時可設 `OPENARM_HAND_VISUAL=skin` 切回原始手部顏色。

```bash
docker compose --profile isaac down
```

## 感知來源與後端

| 設定 | 用途 |
|---|---|
| `OPENARM_PERCEPTION=demo_rgbd` | Isaac 示範場景以 RGB 顏色篩出代理障礙，再從同步深度回推點雲。 |
| `OPENARM_PERCEPTION=mask` | 使用正式外部遮罩 `/perception/hand_mask`、RGB-D 及 CameraInfo。 |
| `OPENARM_PERCEPTION=cloud` | 轉發 `OPENARM_OBSTACLE_CLOUD_TOPIC` 的量測雲，預設取自 `/edgetam_tracker/obstacle_cloud`。可接 EdgeTAM 或有公尺尺度的 RGB／MASt3R 重建。 |
| `OPENARM_PERCEPTION=direct` | 共用風險節點直接訂閱上游發布的 `/perception/obstacles`。 |
| `OPENARM_START_TASK=auto` | Isaac 啟用 ROS 目標入口，硬體啟動觀測模式。 |

模擬容器透過 Compose 專案的 bridge network 與 Fast DDS UDP 交換資料。`OPENARM_ROS_DOMAIN_ID` 預設為 87。接硬體時使用下方的 host network 設定，並將驅動與控制容器設為同一專用 domain。

## 硬體接入順序

1. 先以 `OPENARM_BACKEND=hardware OPENARM_PERCEPTION=cloud` 啟動控制容器，並從實機驅動取得關節回饋、校正 TF 與量測點雲。
2. 用實際 OpenArm 型號和已驗證的硬體驅動提供 MoveIt 規劃與標準 action。確認 TF 能在影像時間轉換到規劃座標，並量測點雲的比例尺與延遲。
3. 將硬體安全、機器人及任務 YAML 放在指定目錄，讓 `OPENARM_CONFIG_DIR` 掛載至 `/runtime-config`。啟用任務前設定 `OPENARM_{SAFETY,ROBOT,TASK}_CONFIG` 為各自的容器內路徑。用運動回饋驗證控制點、限位與停機動作。

```bash
docker compose build control
OPENARM_BACKEND=hardware OPENARM_PERCEPTION=cloud \
  docker compose -f compose.yaml -f docker/hardware.compose.yaml up -d control
```

硬體運動取決於現場的 OpenArm 驅動、相機校正、規劃場景和停機介面。相關 topic 與 action 可在 Isaac 先驗證，實體動作仍需在目標設備驗收。

## 歷史執行入口

Gazebo Classic、Koch、Isaac LAN viewer、Tidybot 和外部模型設定見 [歷史操作手冊](history/2026-10-15-host-workflows.md) 與 [程式索引](CODE_MAP.md)。

## 主機端開發

從 repo 根目錄執行。主機需有 ROS 2 Humble、`colcon`、`vcs`，以及對應的 ROS 套件依賴。上游套件下載至 `.openarm-vendor/`，其中參與建置的兩個套件連結至 `src/`。

```bash
source /opt/ros/humble/setup.bash
bash scripts/fetch_openarm_dependencies.sh
colcon list --base-paths src
colcon build --base-paths src
source install/setup.bash
```

如果 checkout 路徑含中文，ROS 2 Humble 的訊息產生器需要使用 ASCII 的建置輸出路徑。從同一個 repo 根目錄執行以下指令即可保留原始碼位置。

```bash
ros_build_dir=$(mktemp -d /tmp/openarm-ros2.XXXXXX)
colcon --log-base "$ros_build_dir/log" build --base-paths src \
  --build-base "$ros_build_dir/build" --install-base "$ros_build_dir/install"
source "$ros_build_dir/install/setup.bash"
```

離線回歸會檢查感知、場景與 ROS 介面。先生成測試所需的上游 URDF，再將 `REPO_CHECK_PYTHON` 指向已安裝感知套件測試依賴的 Python。

```bash
mkdir -p src/openarm_description/generated
xacro src/openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro \
  bimanual:=true no_prefix:=false ros2_control:=false \
  -o src/openarm_description/generated/openarm_v10_bimanual.urdf
python3 -m venv .venv
.venv/bin/python -m pip install -e 'src/realtime_3d_safety_decision[test]'
bash scripts/check_repo.sh
```
