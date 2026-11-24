# Tidybot：物體進入 1 公尺後反向退避 2 公尺

2026-09-30 新增。此節點控制 **tidybot 底盤**，不是 OpenArm 關節規劃器。
本次 24 個離線測試＋1 個隔離 ROS 假底盤測試通過；[驗證紀錄](../results/tidybot_retreat_2026-09-30.md)。
使用點雲判斷前方障礙，並用 `/tidybot/joint_states` 的 `joint_x`、`joint_y`、
`joint_th` 回饋控制位移。ROS 輸出為 `/tidybot/cmd_vel` 的
`geometry_msgs/msg/Twist`，20 Hz；預設只做 dry-run，不建立該 topic 的 publisher。

## 行為

- 車體座標為 x 前、y 左、z 上。至少 3 張不同時間戳的點雲，確認前半平面內
  有距離底盤原點不超過 **1 m** 的障礙，才觸發。這是距離門檻，不估計物體靠近速度。
- 用支援點數足夠的 XY 網格找最近障礙方向，排除設定高度帶以外與底盤半徑內的點。
  它不是人手分類器；桌子、牆、OpenArm 自身也可能觸發，須先核對 ROI／self-filter。
- 觸發時把反方向轉到世界座標並固定目標：`target = start - 2 m * obstacle_direction`。
  車體旋轉時會重新轉換速度，保持同一世界目標，不隨障礙抖動重設起點。
- 預設最大命令速度 **0.30 m/s**（現場 Isaac 約以三分之一即時速度運行），靠近目標減速，目標誤差 **3 cm** 內送零速並進入 `DONE`。
  因時間步、延遲與控制器動態，不能宣稱數學上精確 2.000 m。
- `DONE` 不自動再退 2 m。撤走障礙、確認新鮮且有效的前方清空點雲後，顯式 reset 才重新待命。
- 點雲超過 1.5 s、位置回饋超過 0.5 s、沒有進展 3 s、整次超過 30 s、時間戳倒退、
  座標跳變或偏離路線都會鎖定 `FAULT` 並要求停止。重播同一 stamp 不刷新 watchdog。
  模擬 reset 導致時間回退時，先確認場景，再重啟節點。

**這是隔離模擬中的單次退避反射，不是可保證無碰撞的導航規劃器。**
前方相機看不到完整後方／側方退路，執行前須在 Isaac 確認整條 2 m 路徑（含車身與手臂）
有空間；尚未整合全向掃描、動態退路障礙規劃或軌跡碰撞檢查。

## 現場已知／待確認

9/30 已由 Isaac 場景與控制圖確認：

- `/tidybot/cmd_vel` 為 `Twist`，有 `_World_TidybotBaseControl_SubscribeTwist` subscriber。
- `linear.x` 是車頭方向、`linear.y` 是車體左方、`angular.z` 是逆時針；底盤可 X/Y 平移。
  `PlanarVelocity` 依 `joint_th` 轉成世界速度並積分到 `joint_x`、`joint_y`、`joint_th`
  的位置命令。USD `metersPerUnit = 1`，位置與速度單位分別為 m 與 m/s。
- Isaac 接收端在最後一筆新訊息 **0.5 秒**後停止積分並保持當前姿態；Stop/Play 後也不重播舊命令。
- `camera_front` 相對旋轉後的底盤座標位於 `(0.1000000, 0.0102028, 1.3149987) m`，
  朝底盤 +X 並向下俯視 20°。設定檔已填入 optical frame 到 base frame 的剛體轉換。
- domain 0 只看到本節點以外零個 cmd_vel publisher，並只有指定 Isaac subscriber；本功能限定隔離模擬。
- 點雲 frame 為 `isaac_front_estimated_optical_frame`，目前仍是 RGB 估計深度。
  其 stamp 是牆鐘、joint stamp 是模擬時間，節點分別檢查遞增並使用本機接收時間判斷斷流，
  不直接相減兩者。**單目估計點雲的公尺尺度仍未校正；USD 的公尺單位不能證明 RGB
  深度模型輸出的 1.0 就是 1 m。**因此目前仍不能啟用 1 m 觸發與 2 m 實際退避。
- 執行前最後還要在當下場景目視確認反方向完整 2 m 退路（含車身與手臂）淨空。

## 1. 開啟 GUI 與預覽節點

先依歷史主機操作手冊啟動 GUI、保持 Isaac Play。在另一個終端從 repo 根目錄進入感知套件。[歷史主機操作手冊](history/2026-10-15-host-workflows.md)

```bash
cd src/realtime_3d_safety_decision
bash scripts/run_tidybot_retreat.sh
```

dry-run 會讀取點雲與關節、計算擬發送速度，僅在
`/realtime_safety/tidybot_retreat/status` 發布 JSON，**連零速都不會送到 cmd_vel**，
所以不會與目前鍵盤控制互相干擾。GUI 本身保持原本影像／點雲顯示，未新增控制按鈕。

狀態監看（再開一個終端）：

```bash
cd src/realtime_3d_safety_decision
source scripts/isaac_ros_env.sh
ros2 topic echo /realtime_safety/tidybot_retreat/status std_msgs/msg/String --no-daemon
```

`WAITING` 表示資料未到或過期；`IDLE` 等待門檻；`MOVING` 計算退避；`DONE` 到達；
`FAULT` 表示鎖定停止。dry-run 不會真的移動，觸發後若不更新位置會因 no progress 進入
`FAULT`，這是正常的 watchdog 行為。不要把未校正預覽的方向／距離當驗收。

## 2. 核對設定後才啟用速度輸出

設定檔位於感知套件的 `configs/`。[tidybot_retreat.yaml](../src/realtime_3d_safety_decision/configs/tidybot_retreat.yaml)

1. `cloud_to_base` 已依現場 USD 外參填妥；矩陣定義為
   `p_base = R * (scale * p_cloud) + translation_m`。仍須用已知距離或 Isaac metric depth
   校正 `cloud_scale_to_m`，並將 `metric_scale_verified` 改為 `true`。單一比例無法修復
   單眼深度的局部形變，正式驗收優先改由 Isaac 發布 Depth／PointCloud2。
2. 底盤 body-frame Twist、全向平移、世界位置回饋與 0.5 秒接收端 watchdog 已確認，
   對應 verification 已設為 `true`。
3. 確認無其他鍵盤／控制節點發布同一 cmd_vel，且只有指定模擬器訂閱。
   節點會檢查唯一 publisher／subscriber；這不能代替確認實際端點身份。
4. 每次測試前確認空曠的 2 m 退路，再將 `retreat_corridor_verified_clear` 改為 `true`。
   不要為了跳過檢查直接修改沒有證據的 verification。

全部確認後，停止 dry-run，再執行：

```bash
bash scripts/run_tidybot_retreat.sh --execute
```

未確認設定時命令會直接拒絕，不建立控制節點。DDS 啟動發現尚未完成也可能暫時造成
唯一端點檢查失敗；待 endpoints 正常、場景 clear 後 reset。

若操作者明確接受未校正 RGB 尺度與未檢查退路，只能在已驗證的隔離模擬使用：

```bash
bash scripts/run_tidybot_retreat.sh --execute --accept-uncalibrated-simulation
```

狀態會明確標記 `execute_uncalibrated_sim`，原始 verification 仍保持 false，不會偽裝成
已完成校正。Isaac Stop 會使資料或模擬時間中斷，控制器依設計鎖定 `FAULT`；再次 Play
後須停止並重跑上述命令，避免自動重播停止前的退避目標。

## 3. 停止、重新待命與驗收

已 source 同一 Isaac ROS 環境的終端：

```bash
# 鎖定停止；execute 模式持續送零速
ros2 service call /tidybot_retreat/stop std_srvs/srv/Trigger '{}'

# 移走障礙到 1.3 m 外且資料正常後，重新待命
ros2 service call /tidybot_retreat/reset std_srvs/srv/Trigger '{}'
```

`Ctrl+C`／SIGTERM 嘗試送最後一筆零速；網路中斷或 SIGKILL 時仍依賴接收端 watchdog。
測試先在空曠隔離模擬場景確認方向與單位，再驗證前／前左／前右物體、底盤已有 yaw、
單次 2 m 後不重複觸發、斷流停止、取消、競爭 publisher 與卡住不動。
保存起終點、世界位移、速度命令、狀態與 Isaac 錄影。本次尚未完成真實 Isaac 位移驗收。

離線測試：

```bash
.venv/bin/python -m pytest -q tests/test_tidybot_retreat.py
```

隔離 ROS 假底盤測試（程式強制 domain 179、localhost-only，不使用 domain 0 的 Isaac）：

```bash
source /opt/ros/humble/setup.bash
RUN_TIDYBOT_ROS_TEST=1 .venv/bin/python -m pytest -q tests/test_tidybot_retreat_ros.py
```
