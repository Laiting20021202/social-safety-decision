# Isaac Sim LAN：網頁關節控制與點雲

目前進度入口：[主 README](../README.md)／[Isaac Sim 交接清單](ISAAC_SIM_HANDOFF.md)。此 LAN viewer 尚未接通 Safety Decision 迴避。

> 2026-09-17 更新：收到非預期移動回報後，已停止本機 GUI／控制程序，並移除啟動腳本預設的 `--isaac-joint-control`。下列關節控制說明與測試是先前紀錄；目前預設啟動僅接收相機、生成及發布點雲，不建立 `/joint_command` publisher。異常原因尚未確認，請勿重新啟用控制進行移動測試。

在專案根目錄啟動：

```bash
bash scripts/run_isaac_gui.sh
```

開啟 <http://localhost:8080/>。其他同網段電腦使用這台電腦目前的 LAN IP；2026-09-17 為 <http://192.168.0.16:8080/>。

9/17 曾以暫時的 systemd user unit 管理背景程序；**9/22 檢查為 inactive，8080 無 listener**。以下是查詢／停止歷史 unit 的命令，unit 不存在時也可能顯示 not found：

```bash
systemctl --user status isaac-lan-gui.service
journalctl --user -u isaac-lan-gui.service -f
systemctl --user stop isaac-lan-gui.service
```

手動啟動前先停止既有 GUI，避免佔用 8080。此 unit 不設定開機自動啟動；重開機後使用上述啟動腳本即可。

## ROS 連線

`scripts/isaac_ros_env.sh` 使用 Humble、domain 0、Fast DDS UDPv4，以及 LAN multicast discovery，沒有綁定遠端 IP。每次執行會依目前網路路由選擇本機 IPv4，產生對應的 Fast DDS 設定；本機明確開啟 unicast discovery listener，以接收其他 participant 的回覆。

其他 ROS 終端機先執行：

```bash
source scripts/isaac_ros_env.sh
ros2 daemon stop
ros2 topic list -t --no-daemon --spin-time 10
ros2 topic echo /clock --once --no-daemon
```

不要在啟動 Isaac GUI 之後再 source `scripts/ros2_lan_env.sh`；該舊腳本會切換至 domain 42 / Cyclone DDS。

## 歷史關節面板與目前影像操作

下列關節面板僅在明確使用 `--isaac-joint-control` 時建立，預設看不到。異常尚未釐清前勿啟用；相關測試只驗證程式條件，沒有證明 runtime 控制問題已修復。

- **Isaac Sim 關節控制**：選擇左／右臂的 14 個旋轉關節之一，設定 0.005–0.05 rad 步進，再按 `−` 或 `＋`。指令只包含選取關節，透過 `/joint_command` 發送 `sensor_msgs/msg/JointState`。
- 每次發送前須有新鮮的 OpenArm `/joint_states` 回饋，且唯一的 subscriber 必須是 Isaac JointControlGraph。只包含其他手部裝置名稱的 `/joint_states` 會被忽略。
- 目標受專案的 OpenArm URDF 關節限制約束。送出後等回饋抵達；尚待回饋時按鈕停用。畫面分別顯示目前位置與最近指令結果。
- **Camera Input**：顯示前方相機的 JPEG 串流。
- **點雲生成方法**：預設使用本機 GPU 執行 Video Depth Anything，主畫面顯示即時彩色點雲。可用滑鼠旋轉、平移、縮放。

這個面板提供模擬關節步進，尚未加入底盤／夾爪控制或 MoveIt 避障規劃。

## 目前輸入與輸出

9/17 三路相機的 `/camera/{front,left,right}/color/image_raw/compressed` 均曾收到完整 JPEG 訊息；GUI 選用 front。當時三路 `camera_info`、`/clock`、OpenArm `/joint_states`、`/joint_command` subscriber 均可發現。9/22 未重新進行遠端 ROS／影像／控制驗證。

本機將前方相機影像與 `configs/isaac_lan.yaml` 中記錄的相機內參用於深度估計與反投影，發布：

```text
/realtime_safety/pointcloud [sensor_msgs/msg/PointCloud2]
frame_id: isaac_front_estimated_optical_frame
axes: x-right / y-down / z-forward
```

目前沒有發現 aligned depth topic。點雲是 RGB 模型估計，尺度未用模擬深度或實測參考校正；frame 名稱也明確表示估計結果。未宣稱相機與機器人座標已對齊，因此不顯示假定配準的 URDF 或距離避障結果。

## 2026-09-17 驗證

- 本機 IPv4 為 `.16`，Isaac Sim / JPEG relay 當時來源為 `.18`，multicast 模式正常發現。
- 獨立訂閱在 25.06 秒收到 124 筆點雲，每筆 12,000 點，約 5 Hz。
- 瀏覽器實際點擊 `＋` / `−`，左臂第一關節從 0 移動至約 0.01 rad 再返回；獨立 ROS observer 確認位置變化。最後另送原始 0 rad 目標，確認復位。
- GUI HTTP / WebSocket 正常，瀏覽器無 JavaScript page error；已確認 RGB 與點雲畫面。
- 控制狀態、GUI 與點雲相關測試通過。詳見 `results/isaac_lan_readiness_2026-09-17.json`。
