# EdgeTAM + 3D Point Cloud Tracking + RGB-D Fusion：既有系統分析

日期：2026-07-31

本文件是實作前的 repository、ROS 介面、執行環境與官方 EdgeTAM API
盤點。所有「目前已有」的敘述均來自 repository 或本機唯讀檢查；沒有把尚未
安裝、尚未接線或尚未測試的功能寫成已完成。

## 1. 摘要

目前 repository 是 Python 3.10 setuptools 應用，不是 colcon/ament ROS 2
package。主要資料源是 RGB 影片、USB camera、MJPEG 或 ROS Image；3D 幾何由
Video Depth Anything、Depth Anything 或 St4RTrack 估測，再以 YOLO mask 取出
障礙物點。ROS 2 目前只負責 RGB 訂閱與 PointCloud2/JSON 發布。

目前沒有：

- ROS Depth Image、CameraInfo 或 PointCloud2 輸入。
- message_filters RGB-D 同步。
- TF2、`/joint_states`、`/tf` 或 `/tf_static` 整合。
- URDF/MoveIt link collision geometry self-filter。
- 完整參數化的 workspace crop、plane removal、DBSCAN/Euclidean clustering。
- 真正一對一 3D Hungarian association 與完整 track lifecycle。
- OBB、真正的 nearest surface point、mask/point-cloud quality 或 re-prompt。
- EdgeTAM source、checkpoint 或 Python package。

因此新 pipeline 會整合到現有 `realtime_safety` Python package，保留 legacy
YOLO pipeline 作 rollback，並新增 ROS 2 ament wrapper、訊息、launch、config
與純演算法模組。新 pipeline 不 import 或呼叫 Ultralytics/YOLO。

## 2. 作業系統、ROS、Python 與 GPU

| 項目 | 實際盤點 |
|---|---|
| OS | Ubuntu 22.04.5 LTS, x86_64, kernel 6.8.0-136-generic |
| ROS | ROS 2 Humble；`ROS_DOMAIN_ID=42`、`rmw_cyclonedds_cpp`、`ROS_LOCALHOST_ONLY=0` |
| 專案 | Python setuptools editable package，Python >= 3.10 |
| Python | 3.10.12 |
| Build tools | colcon 0.20.1、ament_cmake、CMake 3.22.1、GCC/G++ 11.4 |
| GPU | NVIDIA GeForce RTX 4060 Ti 16 GB，compute capability 8.9 |
| Driver | 550.163.01 |
| VRAM（盤點時） | total/free/used = 16380/14690/1379 MiB |
| CUDA toolkit | 12.2 與 12.6；`/usr/local/cuda` 指向 12.6，nvcc 未加入 PATH |
| PyTorch | 2.9.0+cu128，CUDA available，bf16 supported |
| torchvision | 0.24.0+cu128 |
| cuDNN | 9.10.2 |
| numpy/OpenCV/scipy | 1.26.4 / 4.10.0 / 1.13.1 |

`.venv` 設定了 `include-system-site-packages=true`，會同時讀取 ROS、user-site 與
virtualenv packages。`pip check` 已有既存的 torch/torchaudio、numpy/OpenCV、
numba 等版本衝突。因此 EdgeTAM setup 不應直接升級這個環境中的 torch；腳本會
先做版本檢查、固定 source commit，並要求顯式 opt-in 才安裝 dependency。

本機已有 `message_filters`、`tf2_ros`、`tf2_sensor_msgs`、`cv_bridge`、
`image_geometry`、MoveIt 2.5.9 與 PCL dev 1.12.1；沒有 `pcl_ros` 或 ROS
`robot_self_filter` package。新核心因此使用 numpy/scipy，ROS adapter 不強制
依賴 Open3D、PCL Python bindings 或 scikit-learn。

## 3. 專案形態

- Build backend：`pyproject.toml` 的 `setuptools.build_meta`。
- Python package：`realtime_safety`。
- 目前無 `package.xml`、`CMakeLists.txt`、ROS launch 或自訂 message。
- ROS bridge 使用 lazy imports；`rclpy` 不在 pip dependencies，而由
  `/opt/ros/humble` 提供。
- `scripts/run_koch_stream.sh` 固定 source ROS 2 Humble，並啟動現有
  RGB/learned-depth/YOLO pipeline。

新實作採用同一 repository 的 ament_cmake + ament_cmake_python wrapper，不移動
或刪除既有 Python package。如此可直接重用 `realtime_safety`，又可生成
`TrackedObstacle` messages 與提供 `ros2 launch`。

## 4. 目前 ROS topics 與 frames

以下是程式與 config 中能確定的介面，不代表盤點當下所有 topic 都有 live
publisher。

| 用途 | 現有值 | 狀態 |
|---|---|---|
| RGB 輸入 | `$CAMERA_INPUT_TOPIC`（預設 `/rgbd/color/image_raw`） | 可設定為任意標準 ROS 2 Image topic |
| 本機 RGB preview | `/realtime_safety/camera/image_raw` | `sensor_msgs/Image` |
| Depth Image | 未找到 | 尚無 subscriber |
| CameraInfo | 未找到 subscriber | Koch config 註明目前 `K` 全零 |
| PointCloud2 input | 未找到 | 現有程式只有 publisher/measurement subscriber |
| 全場景 cloud output | `/realtime_safety/pointcloud` | `sensor_msgs/PointCloud2` |
| legacy obstacle cloud | `/realtime_safety/yolo_obstacles/pointcloud` | 新 pipeline 將提供相容 alias；不可與 legacy 同時發布 |
| arm/obstacle relation | `/realtime_safety/arm_obstacle_relationships` | `std_msgs/String` JSON schema v1 |
| `/joint_states` | 未找到 | 尚無 subscriber |
| `/tf`、`/tf_static` | 未找到 | 尚無 TF2 integration |
| 現有 cloud fixed frame | `realtime_safety_frame` | 不是 TF tree 中已驗證的 frame |
| preview image frame | `koch_webcam_optical_frame` | image publisher 預設值 |
| robot base/world frame | 未定義 | GUI `/world/...` 是 Viser path，不是 ROS TF |

Koch profile 因 CameraInfo 無有效內參，目前使用暫定的
`fx=fy=272, cx=159.5, cy=119.5`（320x240）。新系統會優先使用有效
CameraInfo；只有明確配置時才允許這組 fallback，diagnostics 會標記
`calibration_fallback`。

現有 PointCloud2 wire format是 `FLOAT32 x/y/z + UINT32 rgb`、
16 bytes/point，QoS 為 reliable/volatile/keep-last-1。內部座標是
`x-right/y-forward/z-up`；Koch wire mode 只把 z 翻成 down，並沒有 TF
轉換。新系統不得靠改 `frame_id` 假裝完成座標轉換。

## 5. 現有資料流

```text
RGB video / USB camera / MJPEG / ROS Image
  -> bounded LatestQueue（drop oldest）
  -> YOLO segmentation + ByteTrack，或本地 2D Hungarian tracker
  -> Video Depth Anything / Depth Anything / St4RTrack dense pointmap
  -> 2D mask 索引 aligned pointmap
  -> robust depth filtering + voxel sampling
  -> percentile AABB + median center + radius
  -> 3D constant-velocity Kalman filter
  -> obstacle PointCloud2 + arm/obstacle JSON
  -> repository 內 demo safety planner / repository 外 Koch NUC CBF
```

主要檔案：

- `realtime_safety/scheduler.py`：多 worker、latest-only queue、legacy pipeline。
- `realtime_safety/pipeline/pointcloud.py`：單目 depth projection、voxel helper。
- `realtime_safety/pipeline/obstacle_3d.py`：mask-to-pointmap 與簡易 unknown cluster。
- `realtime_safety/pipeline/tracker_3d.py`：以既有 2D track ID 為主的 3D KF。
- `realtime_safety/pipeline/robot_self_filter.py`：Koch 綠色手臂的 2D HSV filter。
- `realtime_safety/ros2_bridge/pointcloud_publisher.py`：PointCloud2 publisher。
- `realtime_safety/ros2_bridge/relationship_publisher.py`：CBF 使用的 JSON schema。

## 6. 現有幾何、追蹤與安全能力

| 能力 | 現況 |
|---|---|
| Voxel downsampling | 有；每 voxel deterministic 留點並限制最大點數 |
| 保留 dense geometry | 有 dense learned-depth pointmap；不是原生 RGB-D cloud |
| Workspace crop | 無通用模組；unknown clustering 有硬編碼範圍 |
| Robot self-filter | 有 Koch 2D HSV filter；無 URDF/link/TF 3D filter |
| Plane removal | 有 RANSAC ground estimator供 planner；未從 obstacle cloud 移除 |
| Clustering | 有 cKDTree radius-connected clustering；未完整參數化 |
| 2D association | 有 Hungarian + IoU/center/class；Koch另可用 ByteTrack |
| 3D tracking | 有 `[x,y,z,vx,vy,vz]` KF；主要依賴 2D ID |
| 3D Hungarian | 無；unknown cluster association 非完整一對一 |
| Track lifecycle | 無 TENTATIVE/CONFIRMED/OCCLUDED/LOST/DELETED |
| AABB | 有 percentile AABB |
| OBB | 無 |
| Velocity/prediction | 有 timestamp-derived velocity 與 3 秒 safety prediction |
| Nearest surface | 無；目前用 `center_distance - radius` 球形近似 |
| Quality fusion | 無 pointcloud/mask quality 與 re-prompt |
| TF projection | 無 3D-to-2D TF + CameraInfo projection |

## 7. 現有 CBF / safety contract

真正的 Koch 機器手臂 CBF、FK、Jacobian 與 motor controller 不在此
repository。此 repo 提供：

1. `/realtime_safety/yolo_obstacles/pointcloud` 的表面幾何。
2. `/realtime_safety/arm_obstacle_relationships` 的 versioned JSON，包含
   track ID、center、velocity、radius、hit/missing、距離與 liveness。
3. `docs/KOCH_NUC_AVOIDANCE_PROMPT.md` 記錄控制端的 CBF/fail-safe contract。

新 pipeline 會：

- 以 `/edgetam_tracker/obstacle_cloud` 發布 generic 障礙點雲。
- 以參數控制是否同時 alias 到既有
  `/realtime_safety/yolo_obstacles/pointcloud`；實作後此alias預設`false`，只
  能在legacy publisher已停止後opt in，否則同一controller input會有兩個來源。
- 新增 rich `TrackedObstacleArray`；真正 CBF repo 尚未在本次 workspace，
  所以不能宣稱已驗證 motor/CBF 行為。
- 不修改legacy relationship JSON schema；新節點本身不發布
  `/realtime_safety/arm_obstacle_relationships`，也不會假造robot arm center。
  本次與控制端唯一直接相容層是上述opt-in PointCloud2 alias。若控制端需要
  dynamic relationship，應另以真實robot state與新message實作subscriber/
  adapter。

## 8. 重要風險

1. **目前 `rgbd` CLI 是 placeholder。** `app.py` 接受
   `--depth-mode rgbd`/`--scale-mode rgbd`，但 scheduler 仍建立
   `MonocularDepthBackend`，且 `scale_mode=="rgbd"` 可令
   `metric_valid=True`。新系統不沿用此路徑；真正 RGB-D 資料缺失時要
   diagnostics/fail closed。
2. **現有 PointCloud2 stamp 是 publication time。** 它不是原始 sensor
   timestamp，不能與另一個 RGB topic做嚴格融合。新 sensor subscriber 保留
   input Header stamp；使用 legacy reconstructed cloud 時預設只能
   point-cloud-only。
3. **`realtime_safety_frame` 沒有 TF contract。** RViz 能顯示不代表 frame
   已轉到 base/world。
4. **現有 self-filter 不是 3D robot geometry filter。** 新 TF link filter
   無有效 link frames 時必須顯示 unavailable，不得假裝成功。實作後因
   repository仍無URDF/link collision geometry，self-filter預設關閉；完成
   link spheres或等效collision filter設定前，輸出cloud可能包含robot本體，
   不能保證是external obstacles only。
5. **桌面可能成為 cluster。** 新 plane removal 預設可配置，且近距離
   fail-safe obstacle 不因低 confidence 被刪除。
6. **legacy 預設會建立 YOLO backend。** 新 launch 使用獨立 node，不進入
   scheduler 的 segmentation 建構路徑。
7. **不可同時讓 legacy 與新 node 發布同一 obstacle alias。** 實作後config、
   node與launch都以`false`為預設；它是人工opt-in切換，不是可以與legacy
   並行的duplicate輸出。
8. 本 repo 沒有 RGB-D rosbag 或 ground truth；不能虛構 live FPS、
   mask IoU 或控制成功率。

## 9. 官方 EdgeTAM 核對結果

官方來源：

- Source：<https://github.com/facebookresearch/EdgeTAM>
- 盤點時 `main`：`7711e012a30a2402c4eaab637bdb00a521302c91`
- 官方 model：<https://huggingface.co/facebook/EdgeTAM>
- checkpoint revision：`14b1b75185fee05a4e4ee1c797b2761d035c7ccf`
- `edgetam.pt` SHA256：
  `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`
- checkpoint size：56,116,523 bytes
- 官方 repository 沒有 release/tag，setup script 因此 pin commit。

官方 PyTorch API（不是自創 class）：

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
state = predictor.init_state(video_path)
frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(
    state,
    frame_idx=...,
    obj_id=...,
    points=...,
    labels=...,
    box=...,
)
frame_idx, object_ids, mask_logits = predictor.add_new_mask(
    state, frame_idx=..., obj_id=..., mask=...
)
for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
    ...
predictor.remove_object(state, obj_id)
predictor.reset_state(state)
```

`SAM2ImagePredictor` 也存在，但只做單張 image prompt，不能取代 video mask
propagation。

已核對的官方限制：

- `init_state` 只接受完整 MP4/MP4 bytes 或數字命名 JPEG directory；沒有
  live ndarray append API。
- 第一次 `propagate_in_video` 後 `tracking_has_started=True`，官方
  `_obj_id_to_idx` 會拒絕新 object ID並丟出 RuntimeError。
- 既有 object 可以在已追蹤 frame re-prompt。
- `remove_object` 存在，但新增 object仍需 reset/rebuild。
- video predictor 回傳 mask logits與 object IDs，沒有在 public return value
  提供獨立「mask confidence」。品質評分不能虛構 model score。

因此 wrapper 會隔離官方 dependency，並使用 bounded rolling JPEG frame
buffer/state manager：

- active track/prompt/mask state由 wrapper 持續保存。
- 新物體、刪除物體或 rolling window更新時，安全地 rebuild官方 inference
  state，並重新加入所有 active confirmed tracks。
- 既有物體優先以先前 mask作上一影格 conditioning；新物體或漂移物體使用
  box + multiple positive points，失敗才嘗試 projection mask，再以 box only
  fallback。
- rebuild與 inference 在 latest-only worker，ROS callback不等待模型。
- 任何 import/load/inference例外都顯式回報；point-cloud obstacle不被清空。

這個 manager 是對官方 public API的封裝，不修改 EdgeTAM核心；代價是官方目前
沒有真正 incremental live API，rebuild latency必須實測，不能先宣稱即時。
有界rolling JPEG window只是把live frames轉接到官方「完整MP4/bytes或數字
JPEG directory」API；window或object set變更時會重建，不等於可append的官方
live state。實作環境另遇到PyTorch 2.9的官方grouped multi-object
non-contiguous `view()`失敗；wrapper不patch核心，而是每個ID各建一個官方
predictor state後合併mask，並以`+independent_state`及WARN/degraded揭露。此
fallback延遲隨object數增加，不可當作原生grouped效能。

實作後的ROS adapter不把延遲mask套到「當下最新」cloud。每個submit sequence
保存該次raw cloud、tracks/clusters、prompts、camera transform、frame/RGB與
geometry stamps、generation及safety publication serial。node先發布
point-cloud-only safety output；Edge result只能在exact identity、age不超過
sensor stale timeout且其後尚無任何較新safety output時，對保存context做
fusion，並以原geometry stamp發布第二筆correction。context淘汰、reset、
frame/stamp mismatch、逾時或較新geometry/prediction已發布時一律丟棄，stale
mask不會倒退或修改latest geometry。這是安全的非同步correction，不是零延遲
同步inference。

## 10. 目標資料流

```text
ROS RGB / Depth / CameraInfo / PointCloud2
  -> approximate sync + timestamp validation
  -> TF2 to tracking frame
  -> optional TF-link robot self-filter
  -> workspace crop
  -> voxel + optional outlier/plane removal
  -> DBSCAN/Euclidean 3D clusters
  -> 3D KF + Hungarian + persistent lifecycle
  -> CameraInfo projection
  -> EdgeTAM box/positive points/projection-mask prompt
  -> asynchronous rolling-window mask propagation
  -> exact saved-context mask and original-resolution cloud/depth fusion
  -> optional same-measurement-stamp correction（沒有較新safety output時）
  -> OBB/AABB/nearest surface/velocity/predictions/confidence
  -> generic topics + legacy obstacle-cloud compatibility alias
```

EdgeTAM unavailable、RGB missing或mask invalid時，幾何路徑仍是：

```text
PointCloud2 -> preprocess -> cluster -> 3D tracker -> obstacle outputs
```

## 11. 新增 node 與 topic interface

Node：`edgetam_pointcloud_tracker_node`

Inputs（全部可配置；空 topic表示該來源未提供）：

- RGB：沿用 `/realtime_safety/camera/image_raw`
- Depth：repository無既有值，預設空
- CameraInfo：repository無已訂閱值，預設空
- PointCloud2：沿用 `/realtime_safety/pointcloud`
- JointState：保留`/joint_states`參數，但實作後node沒有直接subscriber；外部
  `robot_state_publisher`必須先產生link TF
- TF：標準 `/tf`、`/tf_static` 由 TF2 listener使用

Outputs：

- `/edgetam_tracker/obstacles`：
  `realtime_3d_safety_decision/msg/TrackedObstacleArray`
- `/edgetam_tracker/obstacle_cloud`：安全障礙表面 PointCloud2
- 預設關閉、人工opt-in的legacy alias
  `/realtime_safety/yolo_obstacles/pointcloud`
- `/edgetam_tracker/debug_cloud`
- `/edgetam_tracker/debug_image`
- `/edgetam_tracker/markers`
- `/edgetam_tracker/diagnostics`
- `/edgetam_tracker/fps`
- `/edgetam_tracker/latency_ms`

預設 tracking/base frame 暫沿用 `realtime_safety_frame`。camera frame不猜測，
空值時使用 image/cloud Header frame；無有效 TF 時不做錯 frame fusion。
由於self-filter預設關閉，這個generic obstacle cloud在完成robot geometry
設定前可能包含self points；「安全障礙表面」不應被誤讀成已保證external-only。

## 12. 預計修改與新增檔案

ROS package metadata與 messages：

- `CMakeLists.txt`
- `package.xml`
- `msg/TrackedObstacle.msg`
- `msg/TrackedObstacleArray.msg`

核心與 ROS node：

- `realtime_safety/edgetam_tracker/__init__.py`
- `realtime_safety/edgetam_tracker/models.py`
- `realtime_safety/edgetam_tracker/sensor_sync.py`
- `realtime_safety/edgetam_tracker/pointcloud_preprocessor.py`
- `realtime_safety/edgetam_tracker/robot_self_filter.py`
- `realtime_safety/edgetam_tracker/cluster_extractor.py`
- `realtime_safety/edgetam_tracker/pointcloud_tracker.py`
- `realtime_safety/edgetam_tracker/projection_utils.py`
- `realtime_safety/edgetam_tracker/quality.py`
- `realtime_safety/edgetam_tracker/mask_pointcloud_fusion.py`
- `realtime_safety/edgetam_tracker/edgetam_wrapper.py`
- `realtime_safety/edgetam_tracker/tracked_obstacle_node.py`
- `scripts/edgetam_pointcloud_tracker_node`

Runtime assets：

- `config/edgetam_pointcloud_tracker.yaml`
- `launch/edgetam_pointcloud_tracker.launch.py`
- `rviz/edgetam_pointcloud_tracker.rviz`
- `scripts/setup_edgetam.sh`
- `scripts/download_edgetam_checkpoint.sh`

Tests/evaluation：

- `tests/test_edgetam_pointcloud_preprocessor.py`
- `tests/test_edgetam_cluster_tracker.py`
- `tests/test_edgetam_projection_fusion.py`
- `tests/test_edgetam_quality.py`
- `tests/test_edgetam_sensor_sync.py`
- `tests/test_edgetam_wrapper.py`
- `tests/test_edgetam_async_context.py`
- `tools/evaluate_tracker.py`
- `results/edgetam_pointcloud_evaluation.md`

Documentation：

- `docs/edgetam_pointcloud_tracker.md`
- `README.md`（只新增入口與 rollback說明，不刪除 legacy內容）

## 13. 現有 tests 與資料

- 原 README test command：
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q`
- 盤點時 collect-only：91 tests collected。
- 一次既有 full suite在受限 sandbox結果為 83 passed、7 GUI socket failures、
  1 model smoke skipped；GUI failures來自 sandbox禁止 localhost socket，不是
  perception assertion失敗。
- 沒有 tracked rosbag、`.db3`、`.mcap` 或同步 RGB-D sample。
- third_party只有 RGB MP4，不能驗證 RGB-D fusion。
- `sessions` 有舊 safety JSONL/CSV，但不是 sensor replay input。

所以本次會執行：

1. 純 numpy/scipy unit tests。
2. deterministic synthetic RGB-D/point-cloud integration sequence，涵蓋移動、
   交會、遮擋、稀疏、RGB loss、mask drift與突然伸入。
3. ROS package build/import tests。
4. 若 EdgeTAM checkpoint未下載，model smoke明確 skip/unavailable；不虛構
   EdgeTAM FPS。
5. 若沒有 live RGB-D publisher/rosbag，結果報告把 live metrics列為
   `not measured`，並只報實際執行的 synthetic point-cloud core metrics。

## 14. 實作後驗證補記

以上第1–13節保留「修改前」盤點語意。本次實作完成後，固定commit的官方
EdgeTAM source與通過SHA-256核對的checkpoint已安裝，並在RTX 4060 Ti以
CUDA/bf16實際跑過兩個ID、rolling window與ID刪除重建。詳細實測與PyTorch
2.9多物件相容性限制見`results/edgetam_official_smoke.md`及
`docs/edgetam_pointcloud_tracker.md`。同步metric RGB-D、真實TF/URDF、
外部CBF與馬達硬體仍未提供，因此相關accuracy與控制結果仍是N/A。
最終完整pytest結果為`153 passed, 1 skipped`；skip只對應未設定
`SAFETY_SMOKE_VIDEO`的real neural-model/video smoke。另在乾淨ROS 2 Humble
build/install tree完成約一分鐘的colcon build。build/tests成功不會把以下未有
真實資料的項目升格成已驗證。

實作後的功能邊界如下：

- `tracking.enabled: false`會在每個geometry frame前reset tracker，只提供
  frame-local clusters；重複的numeric ID不代表跨frame同一物體，也沒有可依賴
  的velocity、occlusion lifecycle或ID continuity。
- tracker core可選擇接收mask IoU cost，但目前ROS adapter不把非同步mask套到
  下一個sensor frame做association；node使用3D centroid/Mahalanobis、AABB
  size與point count，避免stale 2D資訊造成錯配。
- EdgeTAM async結果只對sequence-keyed exact saved context做fusion。base
  point-cloud safety先發布；只有context未過期且之後沒有較新safety stamp/
  publication，才以相同geometry stamp發布refined correction。舊結果不回寫
  latest geometry，且不是每個EdgeTAM-enabled output都會被refine。debug mask
  overlay使用保存的原RGB/Header並標示correction published/skipped，所以可能
  晚於較新debug image抵達；它不是單調的safety topic。
- 現有point-cloud live snapshot的`8.230 Hz`約為`121.5 ms/frame`，但它停用了
  EdgeTAM；獨立官方API smoke的兩ID independent-state呼叫是`469.938 ms`與
  `219.148 ms`，刪除一個ID後的單IDwarm call是`61.905 ms`。這些不是同一run，
  不能組合成ROS throughput；在serial gate下，慢於新geometry publication的
  mask只能用於quality/re-prompt或被discard。尚無同步RGB-D ROS證據顯示
  obstacle cloud實際發布過refined correction或accuracy有所提升。
  真實run需保存diagnostics的`edge_refined_corrections`、
  `edge_stale_results`與`retained_edge_contexts`；counter為零時不得用API
  smoke代替live fusion證據。
- PointCloud2缺席但仍有有限Depth與有效CameraInfo時，可由Depth建立新的3D
  measurement；RGB不同步則停用mask refinement。所有geometry都暫停時，只在
  fallback delay與sensor stale timeout之間發布既有track的bounded Kalman
  prediction，標示`prediction_only`與`PointCloudQuality.INVALID`，不能產生
  新物體；超時後停止輸出並報ERROR。
- CameraInfo由latest-only獨立subscription提供，不在
  `ApproximateTimeSynchronizer`內。實作後adapter會在每次使用前驗證
  shape/frame，並要求非零stamp與reference在slop內；零stamp只在
  `sync.allow_static_camera_info`允許時作為固定校正。這不是多版本CameraInfo
  queue，快速動態解析度/ROI/內參仍需要後續ATS或version contract。
- projection只使用`CameraInfo.K`的pinhole model，不處理distortion
  coefficients；RGB與Depth/cloud必須事先rectify到同一image geometry。
- 單一cluster消失但current raw geometry仍更新時，已有previous mask的
  `OCCLUDED` track可用GOOD/DEGRADED exact-context mask，從同frame finite raw
  depth/cloud做spatial/depth-gated geometry補點；它保持
  `PointCloudQuality.INVALID`與原`last_measurement_stamp`，confidence不增加、
  uncertainty擴大，不算fresh tracker measurement，也不能建立新track或恢復
  `LOST`。完整geometry-input loss仍只有bounded KF prediction。
- `edgetam.prompt_confirmed_tracks_only`必須為true，false會在startup被拒絕；
  runtime只prompt `CONFIRMED`及已有successful previous mask的`OCCLUDED`，
  `TENTATIVE/LOST`不會成為Edge object。
- self-filter預設disabled；alias預設disabled；新node沒有relationship JSON或
  CBF controller integration。這三者都不能由topic存在或build成功推論成已驗證
  的robot avoidance。
- repository仍沒有同步RGB-D rosbag或ground truth。現有
  `results/ros_live_smoke_2026-07-31.md`是EdgeTAM停用的learned-depth
  PointCloud2 snapshot；`results/edgetam_official_smoke.md`是與ROS/RGB-D分離的
  官方API/CUDA smoke；deterministic synthetic報告只驗證core contract。三者
  不可合併宣稱live EdgeTAM + RGB-D end-to-end FPS、latency、exact-context
  correction rate或accuracy。
