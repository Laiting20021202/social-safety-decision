# Social Safety Decision

OpenArm 的 ROS 2 感知、風險判斷與動態避障研究程式。Docker Compose 負責啟動 Isaac Sim 與 ROS 2 Humble 控制。Isaac 與實體設備使用相同的 ROS 訊息和動作介面，感知端可提供 RGB-D 遮罩或有比例尺的障礙點雲。

## 執行 Isaac Sim

需要 Linux x86-64、Docker Compose、NVIDIA GPU 和 NVIDIA Container Toolkit。依 NVIDIA Omniverse 授權條款選擇 `ACCEPT_EULA`。

```bash
bash docker/build.sh
ACCEPT_EULA=Y docker compose --profile isaac up -d
docker compose exec -T control bash docker/entrypoint.sh python3 docker/verify_isaac.py
```

預設場景以 OpenArm 和程序生成的手部幾何測試 ROS 2 管線，`demo_rgbd` 使用既有的 RGB 顏色篩選與對齊深度。可將感知來源切換成已校正的障礙點雲，供 EdgeTAM、RGB 三維重建或之後的真機介接。

```bash
docker compose --profile isaac down
```

## 文件

| 文件 | 內容 |
|---|---|
| [系統流程圖與 ROS 2 套件](docs/ARCHITECTURE.md) | 容器邊界、套件責任、訊息與動作介面 |
| [Docker 與 Sim-to-Real](docs/DOCKER.md) | 建置、驗收、資料來源設定與真機接入 |
| [程式索引](docs/CODE_MAP.md) | 程式入口、感知模組、ROS 2 節點和實驗工具 |
| [歷史交付](docs/DELIVERABLES.md) | 公開分支、原始操作說明與實驗紀錄 |
| [Docker 驗收](results/docker_isaac_2026-11-24.md) | GPU、感知、規劃及 action 的實測結果 |

現有的 Gazebo、Isaac LAN、Koch 和 Tidybot 主機流程見[歷史操作手冊](docs/history/2026-10-15-host-workflows.md)。
