# 給 Koch NUC Codex：使用穩定障礙關係 Topic 完成避障

請把以下整段貼到 Koch NUC（192.168.0.231）上的 Codex：

```text
你正在操作 Koch NUC（192.168.0.231）的 ROS 2 Humble 控制端。請實際檢查、修改、
測試並驗證避障程式；先不要啟動馬達，只進行 topic、座標、RViz 與 dry-run 控制輸出測試。
保留所有與此任務無關的使用者修改。

感知 Host（192.168.0.234）現在提供：

1. /realtime_safety/arm_obstacle_relationships
   type: std_msgs/msg/String
   QoS: RELIABLE / VOLATILE / KEEP_LAST depth=1
   約 8–12 Hz，即使沒有障礙物也會持續發布。

2. /realtime_safety/yolo_obstacles/pointcloud
   type: sensor_msgs/msg/PointCloud2
   只包含已確認、已排除綠色 Koch 手臂的障礙物點。

3. /realtime_safety/pointcloud
   type: sensor_msgs/msg/PointCloud2
   完整場景點雲，只供視覺化或額外幾何檢查，不可直接當成 YOLO 障礙物。

ROS 環境必須統一：
ROS_DOMAIN_ID=42
ROS_LOCALHOST_ONLY=0
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

/realtime_safety/arm_obstacle_relationships 的 String.data 是 JSON，schema：

{
  "schema": "realtime_safety/arm_obstacle_relationships",
  "schema_version": 1,
  "sequence": 123,
  "perception_sequence": 81,
  "perception_age_sec": 0.04,
  "frame_id": "realtime_safety_frame",
  "coordinate_mode": "camera_y_forward",
  "coordinate_convention": "x_right_y_forward_z_down_m",
  "published_at_unix_sec": 0.0,
  "source_timestamp_sec": 0.0,
  "status": "tracking | no_obstacles | arm_not_localized | perception_stale",
  "arm_valid": true,
  "arm": {
    "center_m": {"x": 0.0, "y": 0.42, "z": -0.07},
    "confidence": 0.95,
    "held_frames": 0,
    "fresh_measurement": true
  },
  "obstacle_count": 1,
  "nearest_obstacle": {
    "track_id": 7,
    "class_name": "person",
    "center_distance_m": 0.38,
    "surface_clearance_m": 0.18
  },
  "obstacles": [{
    "track_id": 7,
    "class_name": "person",
    "obstacle_center_m": {"x": 0.2, "y": 0.7, "z": 0.0},
    "velocity_mps": {"x": 0.0, "y": -0.1, "z": 0.0},
    "radius_m": 0.2,
    "confidence": 0.9,
    "hit_count": 8,
    "missing_count": 0,
    "fresh_measurement": true,
    "motion_state": "static | dynamic",
    "delta_from_arm_m": {"x": 0.2, "y": 0.28, "z": 0.07},
    "center_distance_m": 0.35,
    "planar_distance_m": 0.34,
    "surface_clearance_m": 0.15
  }]
}

座標不可自行猜測或交換：
- x：相機右方
- y：相機前方／深度
- z：相機下方
- 單位：公尺
- arm.center_m、obstacle_center_m、delta_from_arm_m 都在同一個
  realtime_safety_frame。
- center_distance_m 是兩中心的 3D 歐氏距離，剛體外參轉換後數值不變。

請依序完成：

1. 找出 Koch 現有 VAMP／CBF 避障訂閱器、控制迴圈、外參轉換與 RViz 設定。
   不要建立第二套會同時控制馬達的節點。

2. 新增 RELIABLE depth=1 的 std_msgs/String subscriber，解析 JSON，嚴格檢查：
   schema_version == 1、frame_id、coordinate_mode、所有數值 finite。
   用「本機收到 message 的 monotonic time」判斷通訊年齡，不要假設兩台電腦的
   wall clock 完全同步。

3. 建立以 track_id 為 key 的障礙物狀態：
   - fresh_measurement=true 時更新中心、速度、半徑。
   - missing_count > 0 時允許短暫預測／保留，但不得把它當新量測。
   - topic 超過 0.35 秒沒收到：感知逾時，輸出 STOP/hold，不可當成無障礙。
   - perception_age_sec > 0.35、perception_sequence 停止增加，或
     status=perception_stale：推論資料過期，同樣輸出 STOP/hold。
   - status=arm_not_localized 或 arm_valid=false：進入 DEGRADED/STOP。
   - status=no_obstacles 且 topic 新鮮：才可判定目前沒有已確認障礙。

4. 使用控制端已驗證的相機外參，把 arm center、obstacle center、velocity 轉到
   Koch base frame。現有參數若仍是：
   camera_forward_offset=0.600 m
   camera_lateral_offset=0.083 m
   camera_height_offset=0.150 m
   camera_pitch_deg=-8.65
   請保留並先用 RViz/靜態量測驗證正負號；不可只改 frame_id。

5. 避障融合方式：
   - 關係 topic 負責穩定 ID、中心、距離、速度與 liveness。
   - /realtime_safety/yolo_obstacles/pointcloud 負責障礙表面幾何。
   - 依中心最近距離，把點雲 cluster 配對到 track_id。
   - 若關係 topic 仍在短暫 hold、但 PointCloud2 一幀為空，不可立刻刪除障礙；
     使用中心＋radius 的保守球體作短暫 fallback。
   - 完整 /realtime_safety/pointcloud 不可直接輸入障礙 CBF，否則桌面、地板與手臂
     都會被當障礙。

6. CBF 至少使用：
   p = obstacle_center_base - controlled_arm_point_base
   d_safe = obstacle.radius_m + robot_link_radius + safety_margin
   h = dot(p,p) - d_safe*d_safe
   並用 obstacle velocity 與 robot Jacobian 建立 h_dot 約束。
   對所有有效 tracks 建立約束，採最危險者；不要只看 nearest_obstacle summary。
   初始保守門檻可設定：
   - center/surface clearance 進入 0.60 m：降速
   - 進入 0.40 m：CBF 主動介入
   - 進入 0.25 m、topic timeout 或 arm invalid：停止
   門檻做成 ROS parameters，不要寫死在 callback。

7. 手臂實際 FK/TCP 或各 link capsule 若已存在，應作為 controlled_arm_point_base；
   視覺 arm center 用來做獨立距離觀測與 sanity check。若兩者差異持續過大，輸出
   calibration warning 並採較保守距離，不要瞬間放行。

8. 加入防抖：
   - 不因單一 width=0 PointCloud2 清除仍在關係 topic 中的 track。
   - STOP/CBF 解除至少需要連續 5 個安全更新。
   - 對中心只做小幅低通，不可再額外延遲緊急接近事件。
   - sequence 倒退或長時間不增加時視為 publisher restart/stall，清除速度估計並
     保守停止，待連續新資料恢復。

9. Dry-run 驗證，不啟動馬達：

   ros2 topic hz /realtime_safety/arm_obstacle_relationships
   ros2 topic echo --once /realtime_safety/arm_obstacle_relationships
   ros2 topic hz /realtime_safety/yolo_obstacles/pointcloud

   連續記錄至少 20 秒，回報：
   - relationship frame 數與 Hz
   - timeout 次數
   - arm_valid 比例
   - track ID 是否連續
   - min center_distance_m / surface_clearance_m
   - PointCloud2 空幀與非空幀數
   - CBF dry-run 介入、退離、STOP 次數

10. RViz 顯示 transformed arm center、每個 obstacle center、連線、距離文字與
    YOLO PointCloud2，確認座標方向後才能要求使用者批准低速馬達測試。

成功條件：
- 空場景 topic 仍連續更新，控制端判定 no_obstacles 而不是 perception lost。
- 人物短暫漏偵測時，同一 track_id 與 CBF 障礙不會一幀出現、一幀消失。
- 綠色 Koch 手臂不會由 YOLO obstacle cloud 再次成為障礙。
- topic 中斷或 arm localization 失效時 fail-safe STOP。
- 在未取得使用者批准前不啟動 Koch 馬達。

完成後回報修改檔案、subscriber QoS、外參矩陣、CBF 公式、dry-run 數據、
RViz 結果與之後啟動低速測試的明確指令。
```
