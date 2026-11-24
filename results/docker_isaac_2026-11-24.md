# 2026-11-24 Docker／Isaac Sim 驗收

## ROS 2 `src/` 工作區複驗

| 項目 | 實測結果 |
|---|---|
| 建置 | Docker 控制映像與主機 `colcon build --base-paths src` 均建置 13 個套件，其中 11 個為 repo 套件、2 個為固定版本的 OpenArm 上游套件。主機測試在 ASCII 路徑輸出建置結果。Isaac 映像亦建置成功。 |
| Isaac 與 MoveIt | 無障礙場景收到同步 RGB-D、光學 TF、時鐘及關節回饋。MoveIt 讀到工作台碰撞物，`home` 經 `EXECUTING` 到 `HOME_REACHED`。 |
| 障礙與 action | 模擬手部發布後收到非空障礙點雲。雙臂動作同時提交，左臂成功，右臂獨立取消。雙側夾爪 action 可連線。 |
| 硬體觀測設定 | 隔離網路中以合成量測雲達到 `SAFE`，重複時間戳觸發 `EMERGENCY_STOP`。此模式測試未連接實體設備。 |
| 離線回歸 | 感知套件 342 passed、2 skipped，模擬套件 120 passed。跳過神經模型與選用的 Tidybot ROS 測試。Gazebo 生成模型的自體過濾測試通過。 |

## 首次 Docker 整合

| 項目 | 實測結果 |
|---|---|
| 建置 | ROS 2 Humble 控制映像建置 12 個 ROS 套件，包含兩個固定 revision 的 OpenArm 上游套件。Isaac Sim 4.5.0 映像由同一份 checkout 建立。 |
| GPU | NVIDIA GeForce RTX 4060 Ti 16 GB，驅動 550.163.01。 |
| 無障礙 | Compose 專案網路內收到推進的時鐘、同步 RGB／深度／CameraInfo、光學 TF 與雙臂關節。感知監測收到空障礙雲並達到 `SAFE`。 |
| 預覽障礙 | 以 `scenario:perception_preview` 啟動代理手部，ROS 感知輸出非空 `PointCloud2`，保持與感測資料相同的座標和時間基準。 |
| 規劃場景 | MoveIt 查詢確認 `isaac_table` 與 `isaac_yellow_table_edge` 皆存在。 |
| 任務 | ROS 任務節點接收 `home`，經 MoveIt 規劃與 Isaac action 回報 `EXECUTING → HOME_REACHED`。 |
| Action | 重疊送出左右手臂軌跡後取消右臂。左臂 `SUCCEEDED`，右臂 `CANCELED`。左臂最終最大關節誤差為 0.00140 rad。雙邊夾爪 action 可連線。 |
| 來源隔離 | 安全和動態控制節點訂閱量測障礙，沒有模擬真值訂閱或切換來源的訂閱端。 |
| 硬體觀測設定 | 以合成的 ROS 2 測量雲測試共用控制容器。持續提供新時間戳會達到 `SAFE`。重送相同時間戳會觸發 `EMERGENCY_STOP`。未啟動動作節點。 |
| 離線回歸 | 根專案 341 passed、3 skipped。模擬子專案 120 passed。跳過的是神經模型、需另啟的 Tidybot ROS 測試、Gazebo 生成網格測試。 |

驗收涵蓋容器整合、感知格式、模擬規劃與動作介面。完整動態避障指標、真機動作與 MASt3R／EdgeTAM 模型在此組合的實驗，有各自的資料與設備條件。

重現方式見 [Docker 與 Sim-to-Real](../docs/DOCKER.md)。
