# EdgeTAM + 3D Point-cloud Tracker 操作與安全說明

本節點以原生 3D 幾何為安全主路徑，EdgeTAM只做可選的 2D 時序 mask
細化：

```text
PointCloud2，或 Depth + CameraInfo
  -> sensor timestamp檢查 / TF2轉到 tracking frame
  -> optional robot-link sphere self-filter
  -> workspace / voxel / outlier / optional plane preprocessing
  -> DBSCAN或Euclidean 3D clustering
  -> 6D constant-velocity Kalman + Hungarian association
  -> TENTATIVE / CONFIRMED / OCCLUDED / LOST / DELETED lifecycle
  -> AABB / PCA OBB / nearest surface / velocity / future positions
  -> safety obstacle messages與PointCloud2
             |
             + RGB + CameraInfo projection
                 -> EdgeTAM prompt與video propagation
                 -> mask quality gate
                 -> original-resolution cloud/depth geometry refinement
```

新 launch不會啟動YOLO、Ultralytics、ByteTrack或舊
`realtime_safety.scheduler`。既有 viewer/safety profiles仍保留供 rollback，
但不可讓兩條 pipeline同時發布同一個 controller topic。

`compatibility.publish_legacy_obstacle_alias`與launch argument
`publish_legacy_alias`的預設值都是`false`。它們是人工切換控制輸入時使用的
opt-in相容開關，不是第二份安全輸出；若legacy publisher仍在運行，啟用alias會
讓同一controller topic同時出現兩個publisher，來源與stale policy變得不明確。

## 安全邊界

Point cloud是障礙物存在與否的主要證據。EdgeTAM不是獨立的障礙物
detector，也不准因下列問題清空仍有效的3D障礙物：

- EdgeTAM source/checkpoint不存在、load或inference失敗。
- RGB遺失、不同步或過期。
- mask為空、漂移、尺寸不符、depth coverage過低或與cluster/prediction不一致。
- 新track尚未有可用的mask。

這些情況下輸出退回 point-cloud-only geometry，mask quality標成
`UNAVAILABLE`、`DEGRADED`或`INVALID`，並由diagnostics回報原因。mask只有通過
quality gate後，才可在3D AABB、depth與robust-distance gates內細化原始高解析
點雲；mask本身不會生成3D點或取代有效cluster。

### 非同步EdgeTAM的exact-context規則

EdgeTAM不是零延遲的current-frame filter。每個合格RGB/geometry bundle採用以下
發布順序：

1. node先立即發布point-cloud-only safety output，不等待GPU。
2. submit後以wrapper sequence保存該次tracks、clusters、raw cloud、prompts、
   camera transform、RGB/geometry stamps、node context generation與publication
   serial的有界deep-copy context。
3. 結果完成時，只能對同一generation、同一frame index與同一RGB stamp的保存
   context做quality/fusion；context age不得超過
   `sync.sensor_stale_timeout_sec`，而且current Track ID的`first_timestamp`
   必須相同，避免tracker reset後numeric ID重用。
4. 只有至少一個track真的通過mask/depth/spatial gates而設為
   `edge_tam_refined=true`，且base output之後尚未發布任何較新的geometry或
   prediction，才再發布一次refined safety output。correction保留原本的
   geometry Header/stamp，所以subscriber可能看到同一stamp的base與refined兩筆
   訊息；安全順序只會是`t1 base -> t1 refined -> t2`，不會在`t2`後倒退發布
   `t1`。在hand-only模式，即時base geometry也只能包含已通過RGB手部語意與
   3D投影一致性檢查的track；未確認的全場景geometry不會作為障礙物發布。
5. 若context已被queue淘汰、stream/gap reset、frame/stamp不符、逾時，或已有較
   新的safety publication，mask不會套到latest cloud，也不會發布倒退stamp。
   identity/age/context錯誤會在Edge diagnostics顯示degraded/error原因；若只
   是serial gate擋下correction，則debug overlay標成`skipped`並增加
   `edge_stale_results`。通過identity檢查的結果最多更新仍為同一persistent
   track的mask/re-prompt記憶。

因此「EdgeTAM-enabled」不代表每個ROS output都已mask-refined，也不能把model
latency從sensor stamp中扣掉假裝同步。一般async safety correction只重發
`TrackedObstacleArray`、obstacle PointCloud2與markers，不重發`debug_cloud`。
另外保存的exact RGB與原RGB Header會產生mask overlay，status明確寫
`safety_correction=published`或`skipped`。這個debug image保留舊/原始RGB
stamp，可能在較新的prompt-only debug frame後才抵達；它是時間誠實的視覺診斷，
不受safety publication serial gate保證單調，不能當作current safety output。

效能證據尤其不能混用：EdgeTAM停用的learned-depth PointCloud2 snapshot是
`8.230 Hz`（約`121.5 ms/frame`）；另一個獨立官方API smoke中，兩ID
independent-state呼叫分別是`469.938 ms`與`219.148 ms`，刪除一個ID後的單ID
warm call才是`61.905 ms`。它們不是同一個ROS/RGB-D run，不能直接推算throughput；
但在上述「任何較新safety publication都禁止舊correction」規則下，兩ID結果若
慢於geometry cadence，常會只能更新quality/re-prompt或被discard。現有證據沒有
證明live `/edgetam_tracker/obstacle_cloud`曾發布一筆在對應
`TrackedObstacle`中標成`edge_tam_refined=true`的correction，也沒有證明辨識率
或surface accuracy因EdgeTAM提升。

真實run應在`/edgetam_tracker/diagnostics`同時記錄
`edge_refined_corrections`、`edge_stale_results`與
`retained_edge_contexts`（Edge status內對應短key為`refined_corrections`、
`stale_results`、`retained_contexts`）。`edge_refined_corrections=0`時不可把
API smoke的nonempty mask描述成live fused safety geometry。

真正應 fail closed的情況不同：

- PointCloud2/Depth沒有有限且可校正的3D幾何時，不使用RGB mask假造障礙物。
- 來源frame無法轉換到`frames.tracking_frame`時，不以更改`frame_id`冒充TF；
  `safety.tf_failure_policy: hold`表示只允許保留既有有效狀態，資料超過
  `sync.sensor_stale_timeout_sec`後停止把它當新measurement並發出ERROR
  diagnostics。下游controller仍必須有topic timeout/STOP policy。
- 啟用self-filter但任一link TF/geometry不可用，且
  `self_filter.fail_closed: true`時，保留輸入點而不發布「已安全過濾」的空場景，
  同時回報fail-closed狀態。
- `PointCloudQuality.INVALID`不會被低品質mask升格；近場有限點則不得只因
  confidence低而刪除。

本repository沒有真正的Koch CBF、FK/Jacobian或motor controller；只能保留
opt-in legacy obstacle-cloud alias，不能宣稱已驗證馬達避障。新node也不發布
`/realtime_safety/arm_obstacle_relationships` JSON；該schema與publisher只
存在於保留的legacy pipeline。

## 已核對的執行環境

2026-07-31唯讀盤點的主機如下；這是相容性資訊，不是performance
benchmark：

| 項目 | 盤點值 |
|---|---|
| OS / architecture | Ubuntu 22.04.5 LTS / x86_64 |
| ROS 2 | Humble；Cyclone DDS；`ROS_DOMAIN_ID=42` |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 4060 Ti 16 GB，compute capability 8.9 |
| Driver / CUDA toolkit | 550.163.01；12.2與12.6，`/usr/local/cuda`指向12.6 |
| PyTorch / torchvision | 2.9.0+cu128 / 0.24.0+cu128 |
| numpy / OpenCV / scipy | 1.26.4 / 4.10.0 / 1.13.1 |
| EdgeTAM source/checkpoint | 唯讀盤點時未安裝 / 未下載；本次實作後已安裝並驗證，見下文 |
| RGB-D bag / ground truth | repository內沒有 |

`.venv`目前包含system site packages，且盤點時`pip check`已有既存版本衝突。
不要為EdgeTAM直接升級system Python或任意更換torch；使用下節的固定版本腳本。

本次實作後已在`.venv`安裝固定commit的editable EdgeTAM source，並下載通過
size/SHA-256核對的checkpoint。官方模型已在RTX 4060 Ti以CUDA/bf16實際載入與
推論；這是後續驗證狀態，不應回寫成「實作前盤點時已存在」。

## 官方 EdgeTAM pin與API

整合只使用[官方EdgeTAM repository](https://github.com/facebookresearch/EdgeTAM)
的public API，沒有自創predictor方法。

| 資產 | 固定值 |
|---|---|
| Source commit | `7711e012a30a2402c4eaab637bdb00a521302c91` |
| Builder | `sam2.build_sam.build_sam2_video_predictor` |
| Model config | `configs/edgetam.yaml`，官方1024×1024 config |
| Checkpoint revision | `14b1b75185fee05a4e4ee1c797b2761d035c7ccf` |
| `edgetam.pt` size | 56,116,523 bytes |
| SHA-256 | `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df` |

可直接核對[pinned source
commit](https://github.com/facebookresearch/EdgeTAM/commit/7711e012a30a2402c4eaab637bdb00a521302c91)
與[pinned checkpoint
object](https://huggingface.co/facebook/EdgeTAM/resolve/14b1b75185fee05a4e4ee1c797b2761d035c7ccf/edgetam.pt)。

核對過的呼叫面：

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "configs/edgetam.yaml",
    "/path/to/edgetam.pt",
    device="cuda",
)
state = predictor.init_state(video_path="/numeric/jpeg/directory")
predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=0,
    obj_id=track_id,
    points=points_xy,
    labels=point_labels,
    box=box_xyxy,
)
predictor.add_new_mask(
    inference_state=state,
    frame_idx=0,
    obj_id=track_id,
    mask=projection_mask,
)
for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
    ...
predictor.remove_object(state, track_id)
predictor.reset_state(state)
```

官方video predictor的`init_state`接受完整MP4/bytes或數字命名JPEG
directory，沒有live ndarray append API；第一次propagation後也不能直接加入
新的object ID。因此`EdgeTAMWrapper`使用有界rolling JPEG window，object set或
window改變時重建state，並重新加入所有active confirmed track。prompt fallback
順序是：

1. 3D projection box + multiple positive points（若有negative points也一併用）。
2. 3D projection mask。
3. box only。

inference在capacity=1的latest-only worker執行；舊pending job會被新job取代。
官方public video return只有object IDs與mask logits，沒有獨立mask confidence，
所以系統不會虛構model score。rolling rebuild的latency必須在真實stream實測。
這個rolling window是wrapper為官方「完整影片/數字JPEG目錄」介面所做的有界
轉接，不代表官方predictor具有可append的永久live state；window滑動、object
集合改變、stream reset或geometry gap recovery都可能觸發重建。wrapper持有的
previous mask也只是下一次prompt/品質檢查的狀態，不是新的3D measurement。

目前環境的PyTorch 2.9.0對官方多物件batch路徑揭露一個上游相容性問題：
`sam2/modeling/perceiver.py`把expanded、non-contiguous tensor以`view()`重排而
失敗。專案的setup會在固定commit上套用可審核的單行`reshape()` patch，
多個ID因此共用同一個grouped predictor state。wrapper仍保留
`+independent_state`逐ID fallback作為異常保護；任何重試失敗仍是明確
ERROR，絕不回傳空mask成功。

該次smoke的optional upstream `_C` CUDA extension不可用，因此官方程式略過
mask-hole post-processing並輸出warning；core propagation仍有執行，但邊界品質
可能不同於成功建置extension的環境。這項API smoke不構成mask accuracy驗證。
主機雖有`/usr/local/cuda -> cuda-12.6`且可直接找到12.6 toolkit，但`nvcc`不在
執行時`PATH`，所以本次setup採`SAM2_BUILD_CUDA=0`；文件不宣稱extension曾成功
build。PyTorch runtime仍是`2.9.0+cu128`。

## 安裝與驗證 EdgeTAM

先準備project virtual environment與相容的torch/torchvision，再執行：

```bash
cd /path/to/realtime_3d_safety_decision
bash scripts/setup_edgetam.sh
bash scripts/download_edgetam_checkpoint.sh
```

`setup_edgetam.sh`會：

- 拒絕把EdgeTAM安裝到system Python。
- clone official repository到`third_party/EdgeTAM`並checkout上述detached
  commit；套用`patches/edgetam_pytorch29_grouped_objects.patch`；除該精確patch
  以外的local changes會使setup停止。
- 檢查`torch>=2.3`與`torchvision>=0.18`，安裝`timm==1.0.15`及
  `eva-decord>=0.6.1`，再editable-install固定source。
- 找得到`nvcc`時預設建置CUDA extension，否則使用`SAM2_BUILD_CUDA=0`；
  可在執行前顯式設定。
- 最後import官方builder並再次核對Git commit。

下載腳本使用固定Hugging Face revision，先下載到temporary file，核對size與
SHA-256後才atomic rename。若目標已有錯誤checksum，腳本會拒絕覆寫，應先人工
確認檔案來源再移除或改用新的輸出路徑。

可做唯讀核對：

```bash
git -C third_party/EdgeTAM rev-parse HEAD
sha256sum models/edgetam/edgetam.pt
.venv/bin/python -c \
  'from sam2.build_sam import build_sam2_video_predictor; print(build_sam2_video_predictor)'

# Actual official checkpoint / CUDA / two-ID / deletion smoke.
.venv/bin/python tools/smoke_edgetam_official.py
```

RTX 4060 Ti支援bf16，所以`edgetam.precision: auto`預期選bf16；實際resolved
device/precision仍應以node diagnostics為準。CPU會使用fp32；bf16/fp16只允許
CUDA。

## ROS 2 build

```bash
cd /path/to/realtime_3d_safety_decision
source /opt/ros/humble/setup.bash
source .venv/bin/activate
colcon build --symlink-install \
  --packages-select realtime_3d_safety_decision
source install/setup.bash
```

package使用`ament_cmake`、`ament_cmake_python`與
`rosidl_generate_interfaces`生成：

- `realtime_3d_safety_decision/msg/TrackedObstacle`
- `realtime_3d_safety_decision/msg/TrackedObstacleArray`

純演算法tests/evaluation不需要ROS graph：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_edgetam_async_context.py \
  tests/test_edgetam_pointcloud_preprocessor.py \
  tests/test_edgetam_cluster_tracker.py \
  tests/test_edgetam_projection_fusion.py \
  tests/test_edgetam_quality.py \
  tests/test_edgetam_sensor_sync.py \
  tests/test_edgetam_wrapper.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python \
  tools/evaluate_tracker.py

# Requires the built/sourced ROS package and a DDS-capable shell.
python3 tools/ros_tracker_integration_smoke.py

# Requires the official source/checkpoint and CUDA.
.venv/bin/python tools/smoke_edgetam_official.py
```

Final verification另執行完整suite：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q
```

實際結果為`197 passed, 1 skipped`；唯一skip是需要另設
`SAFETY_SMOKE_VIDEO`的real neural-model/video smoke，不是被寫成pass。另以乾淨
ROS 2 Humble build/install tree執行`colcon build`，package約一分鐘完成且成功。
這些結果證明build與測試contract，不取代後文仍為N/A的同步RGB-D accuracy、
live Edge correction rate或CBF/hardware驗收。

## 啟動

目前`koch_lan`的8080整合入口會同時啟動app與EdgeTAM node，
並在UI提供`EdgeTAM + RGB Hand Gate + 3D PointCloud`與
`Legacy YOLO + RGB Hand Gate + 3D PointCloud`兩個手動選項：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
bash scripts/run_koch_stream.sh
```

UI選擇會保持不變；過期或錯誤diagnostics不會自動改選另一模型。
選定輸入過期時，mux保留該模式並讓controller timeout導向STOP，只有
操作者再次使用下拉選單才會切換。狀態欄顯示真實diagnostics、Edge
latency、refined correction counter，並顯示latest-completed exact-RGB mask；
無可信prompt時用當下RGB及`no mask`文字清除過期mask。

`koch_lan`固定相機啟動後先做12 frames warmup與16 frames空場background
calibration；UI顯示橘色校正狀態時必須保持工作區淨空。完成後，木板、桌緣與
靜止構件的3D baseline會先從PointCloud2扣除，只有新出現且離baseline超過
`0.03 m`的幾何才進入DBSCAN與EdgeTAM prompt。若相機或固定場景移動，必須重啟
服務並重新空場校正。

先以不碰controller、不依賴EdgeTAM的模式檢查point cloud、TF與RViz：

```bash
ros2 launch realtime_3d_safety_decision \
  edgetam_pointcloud_tracker.launch.py \
  use_edgetam:=false \
  publish_debug:=true \
  publish_legacy_alias:=false
```

確認3D geometry、frames、nearest point、track lifecycle與stale behavior後，才啟用
EdgeTAM：

```bash
ros2 launch realtime_3d_safety_decision \
  edgetam_pointcloud_tracker.launch.py \
  use_edgetam:=true \
  publish_debug:=true \
  publish_legacy_alias:=false
```

要replay真實bag：

```bash
ros2 launch realtime_3d_safety_decision \
  edgetam_pointcloud_tracker.launch.py \
  input_mode:=bag \
  use_sim_time:=true \
  play_bag:=true \
  bag_path:=/absolute/path/to/rgbd_bag \
  bag_rate:=1.0 \
  publish_legacy_alias:=false
```

`input_mode`只是diagnostic label，不會把單目RGB自動變成metric RGB-D。bag至少
要提供可用PointCloud2，或同步Depth +有效CameraInfo；需要projection/fusion時
還要RGB與完整TF。

RViz設定：

```bash
rviz2 -d "$(ros2 pkg prefix realtime_3d_safety_decision \
  --share)/rviz/edgetam_pointcloud_tracker.rviz"
```

## Topics

預設值都在`config/edgetam_pointcloud_tracker.yaml`。空字串表示該sensor未
提供，不是假裝收到資料。

### Inputs

| Parameter | Default topic | Type | 說明 |
|---|---|---|---|
| `topics.pointcloud` | `/realtime_safety/pointcloud` | `sensor_msgs/msg/PointCloud2` | 主要3D來源；保留input Header stamp |
| `topics.rgb_image` | `/realtime_safety/camera/image_raw` | `sensor_msgs/msg/Image` | EdgeTAM/overlay；遺失時走point-cloud-only |
| `topics.depth_image` | empty | `sensor_msgs/msg/Image` | `16UC1` mm或`32FC1` m |
| `topics.camera_info` | `/realtime_safety/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | app以同RGB stamp/frame發布；projection與depth deprojection所需有效K |
| `topics.set_enabled_service` | `/edgetam_tracker/set_enabled` | `std_srvs/srv/SetBool` | runtime只開關EdgeTAM refinement；不關閉point-cloud tracker |
| `topics.joint_states` | `/joint_states` | parameter only；目前node不建立subscription | robot_state_publisher必須由外部訂閱joint vector並產生各link TF，self-filter只查TF centers |
| TF | `/tf`、`/tf_static` | standard TF2 | sensor/camera/link到tracking frame |

`/realtime_safety/pointcloud`是learned-depth publisher；目前app把Image、
CameraInfo與PointCloud2的source/acquisition stamp轉到同一ROS clock domain。
這可讓live approximate sync與projection實際執行，但不是原生RGB-D sensor或
accuracy/標定證據。

### Outputs

`obstacles`、generic obstacle cloud、diagnostics、FPS與latency publishers固定建立；
legacy alias只在opt in時建立。self-filter/debug cloud、debug image與markers則
受`performance.publish_debug_*`/`publish_markers`控制，launch的
`publish_debug`預設為`false`。

| Default topic | Type | 內容 |
|---|---|---|
| `/edgetam_tracker/obstacles` | `TrackedObstacleArray` | ID、state、quality、confidence、measured/filtered centroid、AABB/OBB、velocity、nearest point、future positions、last measurement stamp、prediction-only與uncertainty |
| `/edgetam_tracker/obstacle_cloud` | `sensor_msgs/msg/PointCloud2` | 所有可發布track的障礙表面幾何 |
| `/realtime_safety/yolo_obstacles/pointcloud` | `sensor_msgs/msg/PointCloud2` | 預設不建立publisher的opt-in compatibility alias；名稱保留但新node不執行YOLO |
| `/edgetam_tracker/self_filtered_cloud` | `sensor_msgs/msg/PointCloud2` | tracking-frame中self-filter之後、workspace crop/preprocess之前的檢查點雲；filter停用時不能視為已移除robot或已裁切 |
| `/edgetam_tracker/debug_cloud` | `sensor_msgs/msg/PointCloud2` | 與可發布track障礙表面相同的geometry，以Track ID上色；不是preprocess cloud |
| `/edgetam_tracker/debug_image` | `sensor_msgs/msg/Image` | 無候選或輸出被hold時發布當下RGB與`no mask`狀態；EdgeTAM ready且有prompt時抑制會覆蓋結果的prompt-only frame，只顯示latest-completed exact-RGB mask overlay與correction published/cached狀態 |
| `/edgetam_tracker/markers` | `visualization_msgs/msg/MarkerArray` | OBB、label、nearest point、velocity、predictions |
| `/edgetam_tracker/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | sensor/TF/self-filter/EdgeTAM/fallback與queue狀態 |
| `/edgetam_tracker/fps` | `std_msgs/msg/Float32` | 實際completed point-cloud pipeline rate |
| `/edgetam_tracker/latency_ms` | `std_msgs/msg/Float32` | 實際geometry pipeline latency；延後完成的EdgeTAM latency另在diagnostics，不包含在此數值 |

目前RViz設定中的display label `Self-filtered / Cropped Cloud`沿用早期名稱；實際
topic是上表所述的pre-crop self-filter output，不能由label推論workspace crop已
套用。

安全輸出應使用reliable、volatile、keep-last-1；sensor subscriptions可沿用
sensor-data QoS。所有輸出frame必須真正在`frames.tracking_frame`，stamp保留
對應sensor measurement time；不要用publication time替代同步依據。

## Frames、CameraInfo與座標

- 預設`frames.tracking_frame`與`frames.robot_base_frame`都是既有
  `realtime_safety_frame`，但repository內沒有可證明它們正確的TF tree。真實
  robot應改成已校正的`base_link`/world frame。
- `frames.camera_frame`空白時，以message Header frame為準，不猜名稱。
- 標準ROS optical cloud使用`projection.camera_axis_mode: ros_optical`
  （x right、y down、z forward）。
- 既有Koch wire cloud使用
  `projection.camera_axis_mode: x_right_y_forward_z_down`
  （x right、y forward、z down）。
- cluster `depth_variance` quality feature由
  `clustering.depth_axis`選擇tracking-frame的0=x、1=y或2=z；預設1符合既有
  `x-right/y-forward/z-up` contract。把tracking frame改成慣用`base_link`
  （常見x-forward）時，應同步把此參數改為0，並重新驗證quality thresholds與
  workspace bounds。
- 只有有效的`CameraInfo.K`才能做校正projection/deprojection。目前Koch
  CameraInfo已知可能全零；不要在安全融合中靜默接受。
- projection/deprojection目前是`K`矩陣pinhole運算，不套用`CameraInfo.D`做
  distortion correction；RGB、Depth/cloud與CameraInfo必須已在同一rectified
  image geometry。raw distorted image不能視為已校正fusion input。
- `CameraInfo`使用獨立latest-only subscription，不是
  `ApproximateTimeSynchronizer`成員。adapter在每次使用前另外要求
  width/height與frame ID吻合；非零stamp必須在`sync.slop_sec`內，零stamp則只在
  `sync.allow_static_camera_info: true`時視為固定校正。這能拒絕明顯錯配，但
  latest-only subscription仍不保存多個calibration版本；相機若會快速切換
  resolution/ROI/intrinsics，不能宣稱逐frame CameraInfo ATS同步，應先擴充
  queue/version contract或停用fusion。
- `sync.slop_sec`只允許近似同步；任何bundle仍須通過
  `sync.max_data_age_sec`與timestamp檢查。PointCloud/Depth可在RGB消失後經
  `sync.pointcloud_fallback_delay_sec`進入安全fallback。

## Track與geometry contract

每個cluster保留finite original-resolution points、source indices與可用時的
pixel correspondence。voxel/outlier/plane結果只用於較省成本的clustering；
`pointcloud.use_high_resolution_for_final_geometry: true`時，最終表面再從原始
workspace cloud細化。

`missing_depth_ratio`只在有效2D pixel correspondence存在時估計cluster footprint
內「完全被觀測輪廓包住」的拓撲洞；與輪廓邊界相連的未知區或voxel稀疏不被
當成depth hole。未organized、未校正或無pixel correspondence的PointCloud2
會回報`0.0`（unknown/not measurable），不能把該數字解讀成depth完整。

association是一對一Hungarian assignment，cost可組合：

- centroid / Mahalanobis distance；
- AABB size差；
- point-count差；
- 有效時才使用mask IoU。

最後一項是pure tracker core提供的可選輸入；目前ROS adapter不把非同步
EdgeTAM mask回填到下一個frame的association cost，因為那會混用不同sensor
context。實際node的跨frame assignment目前使用3D centroid/Mahalanobis、AABB
尺寸與point count；EdgeTAM結果在通過同一提交context檢查後只參與mask品質與
幾何細化。

`tracking.enabled: false`（launch為`use_pointcloud_tracking:=false`）不是
另一種persistent tracker。node會在每個geometry frame更新前reset tracker，
因此只輸出該frame的clusters；numeric ID會從1重新配置，相同數字不代表同一
實體，velocity、跨frame lifecycle、遮擋延續與ID穩定性都不可使用。預設
`true`才啟用本節其餘的temporal contract。

輸出state意義：

| State | 意義 |
|---|---|
| `TENTATIVE` | 尚未達`tracking.confirmation_hits`的新track |
| `CONFIRMED` | 有足夠連續3D evidence；近於emergency distance可立即升格 |
| `OCCLUDED` | 短期沒有measurement，使用有界CV prediction |
| `LOST` | 超過短遮擋但仍在retention內，不應無限預測 |
| `DELETED` | pure tracker core的terminal state可存在一個update；目前ROS adapter在發布前濾除它，因此subscriber看到track消失，不會收到一筆`DELETED` obstacle |

`nearest_point`是實際track surface sample中離robot origin最近的點，不是
`center - radius`球形近似。future positions來自6D state；uncertainty隨
covariance與lost age增加。

PointCloud2/Depth暫停但尚未超過stale timeout時，worker會以設定速率發布既有
track的有界Kalman prediction。這類訊息：

- Header stamp是prediction time；
- `last_measurement_stamp`仍是最後一次sensor measurement time；
- `prediction_only=true`，state轉為`OCCLUDED`/`LOST`；
- confidence衰減、uncertainty增加，且point-cloud quality標成`INVALID`；
- 只外推已存在track與其最後表面geometry，不能由RGB/EdgeTAM建立新障礙物；
- 超過`sync.sensor_stale_timeout_sec`後完全停止安全輸出並發布ERROR，而不是
  持續用新timestamp包裝舊點。

這個「geometry gap prediction」不要和sensor synchronizer的depth fallback
混為一談。若PointCloud2缺席但收到有限metric Depth與有效CameraInfo，node可
將該Depth deproject成新的3D measurement；RGB不在`sync.slop_sec`內時則只做
depth/point-cloud追蹤，不做mask fusion。若Depth與PointCloud2都沒有新資料，
才進入上述只維持既有track的有限prediction window。geometry恢復後會清除
舊EdgeTAM rolling/mask記憶並要求重新建立同一sensor context，避免跨gap套用
mask。

若sensor仍送來current raw geometry，但某一已確認物體暫時沒有形成cluster，
3D tracker先把它標成`OCCLUDED`並把point-cloud quality設為`INVALID`。只有已有
previous mask memory的`OCCLUDED` ID會繼續收到EdgeTAM prompt；`TENTATIVE`與
`LOST`不會走這條恢復路徑。GOOD/DEGRADED exact-context mask可在cached predicted
AABB、camera-z depth與robust-distance gates內，從同一frame的finite raw
depth/cloud補入表面點。這是「mask + current depth」的保守geometry
refinement，不是mask-only detection：

- track state、`last_measurement_stamp`與hit count不會變成新3D measurement；
- `PointCloudQuality`保持`INVALID`，confidence不得高於原prediction，
  uncertainty隨miss增加；
- 不能建立新track，也不能讓`LOST` track復活；
- 沒有current finite depth/pixel correspondence、mask不合格，或整個geometry
  input中斷時，仍只剩上述bounded KF prediction，最後依stale timeout停發。

基於此安全政策，`edgetam.prompt_confirmed_tracks_only`必須為`true`；node在
startup明確拒絕`false`，避免TENTATIVE noise成為Edge object。runtime候選固定
為`CONFIRMED`，加上已有successful previous mask memory的`OCCLUDED`；
`TENTATIVE`與`LOST`永遠不prompt。

## 主要參數

完整值與註解請以`config/edgetam_pointcloud_tracker.yaml`為準：

| Group | 重要參數 / 預設 |
|---|---|
| Workspace | x `[-1.5, 1.5]`、y `[0.05, 4.0]`、z `[-2.0, 2.0]` m |
| Background | `koch_lan` warmup `12` + calibration `16` frames；baseline voxel `0.015` m；foreground distance `0.03` m |
| Preprocess | voxel `0.025` m；statistical outlier；`koch_lan`只移除距相機`0.34–0.48 m`且法向接近depth axis的主平面 |
| Cluster | DBSCAN；tolerance `0.08` m；min points `12`；max clusters `64`；depth axis `1` |
| Track | confirmation `3`；association distance `0.35` m；prediction horizons `[0.2, 0.5, 1.0]` s |
| Projection | box padding `8` px；至少6 projected points；5 positive prompts |
| EdgeTAM | CUDA/auto precision；1024×1024；max objects `4`；rolling window `1`；JPEG quality `82` |
| Fusion | erode `1` px；depth gate `0.12` m；min mask/cluster IoU `0.25` |
| Stale safety | max data age `0.35` s；sensor stale timeout `0.75` s |
| Queue | latest-only；maximum queue length `1` |
| Gap prediction | `10 Hz`；只在fallback delay後、stale timeout前 |

參數不是跨相機/場景的保證值。workspace、cluster tolerance、plane policy、
emergency distance、self-filter link spheres與stale timeout必須用真實bag及
已測量的robot geometry重新驗證。

## Robot self-filter

repository沒有URDF、link collision meshes或ROS `robot_self_filter` package，
所以安全預設是`self_filter.enabled: false`，而不是假裝已濾掉robot。
在這個預設下，`/edgetam_tracker/obstacle_cloud`可能包含機器手臂或夾爪本身；
在已校正link spheres或等效URDF collision filter完成設定與驗收前，不能保證
該topic是「external obstacles only」。

目前core接受已轉到tracking frame的link spheres。啟用前，設定等長的
`self_filter.link_frames`與`self_filter.link_radii_m`，padding已另外加入：

```yaml
self_filter.enabled: true
self_filter.link_frames: [base_link, shoulder_link, elbow_link, wrist_link]
self_filter.link_radii_m: [0.10, 0.08, 0.07, 0.06]
self_filter.padding: 0.03
self_filter.fail_closed: true
```

每個radius必須覆蓋該link在所有關節角度下的實際collision envelope。若需要
mesh級精度，應在ROS adapter換成已驗證的URDF collision filter，不可把視覺
HSV arm filter當作3D self-filter。

## Legacy controller與rollback

既有controller cloud是：

```text
/realtime_safety/yolo_obstacles/pointcloud
```

只有確定舊YOLO publisher已停止、且新geometry/frame/stale policy已通過
staging測試後，才可設定：

```bash
ros2 launch realtime_3d_safety_decision \
  edgetam_pointcloud_tracker.launch.py \
  publish_legacy_alias:=true
```

不要以topic名稱中的`yolo`誤判新pipeline仍執行YOLO；這只是controller
compatibility alias。新node不會假造
`/realtime_safety/arm_obstacle_relationships`內的arm center，也沒有修改其
versioned JSON schema；更精確地說，新node完全沒有該JSON publisher或CBF
adapter，只有上述PointCloud2 alias。若controller需要relationship JSON，必須
在外部以新`TrackedObstacleArray`與真實robot state明確實作並驗證。

Rollback：

1. 停止`edgetam_pointcloud_tracker_node`。
2. 確認legacy alias已無publisher。
3. 若只需保留RGB/depth reconstruction而不啟動Edge node，可用
   `EDGETAM_TRACKER_ENABLE=0 bash scripts/run_koch_stream.sh`；此模式不會自動
   恢復legacy YOLO controller output。真正rollback必須明確啟動已驗證的legacy
   service/profile。
4. 用`ros2 topic info ... --verbose`確認controller input只有預期的一個
   publisher。

## 真實驗收清單

合成評估結果在`results/edgetam_pointcloud_evaluation.md`，只證明
deterministic functional behavior。上線前仍須依序完成：

1. 錄製含RGB、metric Depth/organized PointCloud2、有效CameraInfo、TF、
   `/joint_states`的rosbag，且所有sensor stamp同一clock domain。
2. 以已知尺寸/距離標定3D誤差、nearest surface與workspace邊界。
3. 標註moving、crossing、短遮擋、離場/重入、sparse/depth hole、mask drift、
   RGB loss、robot motion與突然伸手案例。
4. 比較point-cloud-only和EdgeTAM-enabled的ID switches、track recall、
   surface distance error、mask IoU與fallback rate。
5. 在目標RTX 4060 Ti量測completed FPS、p50/p95 latency、queue drops與VRAM；
   EdgeTAM rolling-state rebuild另行列出，不能拿單一model forward time代替。
6. 故障注入：拔RGB、拔depth/cloud、停止TF、破壞CameraInfo、移除checkpoint、
   讓mask漂移，確認沒有「空場景等於安全」。
7. 將新cloud接到外部CBF的staging mode，驗證topic timeout、STOP/release
   debounce與motor limits，再進硬體低速測試。

官方checkpoint的離線API latency/VRAM已有一次實測。修正PyTorch 2.9
grouped-object相容層後，`koch_lan`同步生產設定（window `1`、兩個ID）在
RTX 4060 Ti/CUDA/bf16的首次warm-up後為44.8–52.3 ms（約19–22 model
calls/s），高於整體pipeline的12 Hz上限。有效projection且修正
稀疏depth-support gate後，固定下緣候選曾產生DEGRADED mask與34筆
exact-context refined corrections。加入空場baseline後，同一空場實測為
`processed=0, cluster=0, track=0, prompt=0`。尚未取得有人手進入的同步ground
truth，因此hand recall、mask accuracy、RGB-D fusion accuracy、self-filter recall
與CBF成功率仍維持`N/A`。

目前證據必須分開解讀：

- 原`ros_live_smoke_2026-07-31.md`是point-cloud-only snapshot；之後8080 live
  integration已補上同步RGB/CameraInfo並執行EdgeTAM，但仍沒有ground truth。
- `edgetam_official_smoke.md`只驗證官方checkpoint、CUDA/bf16、rolling window、
  grouped multi-ID與ID刪除；它沒有ROS graph、Depth/CameraInfo、TF或mask GT。
- deterministic evaluation只驗證合成core contract，不是相機accuracy或即時
  benchmark。

因此不能把point-cloud snapshot的FPS/latency、固定構件的refined counter與官方
API smoke拼成accuracy benchmark；目前只可報告live inference、background
rejection、mask gate與correction path已運作，不能宣稱已驗證人手辨識率。

## 調參與常見問題

### Live camera要怎麼接

`input_mode:=live`只是diagnostic label；真正介面仍由YAML topics決定。原生
RGB-D相機建議提供對齊的RGB、Depth、CameraInfo與PointCloud2，並把
`projection.camera_axis_mode`設成`ros_optical`。若只提供Depth，node會用有效
CameraInfo deproject；若只提供現有learned-depth PointCloud2，則能做3D追蹤，
但沒有校正pixel correspondence時不啟用mask/cloud融合。

啟動RViz可直接加：

```bash
ros2 launch realtime_3d_safety_decision \
  edgetam_pointcloud_tracker.launch.py \
  use_edgetam:=false \
  publish_debug:=true \
  publish_legacy_alias:=false \
  launch_rviz:=true
```

### GPU記憶體不足

依序降低`edgetam.maximum_objects`與`edgetam.rolling_window_frames`，關閉debug
image/cloud，並考慮`edgetam.offload_video_to_cpu: true`或
`edgetam.offload_state_to_cpu: true`。官方pinned config固定1024×1024；目前
wrapper會拒絕假裝其他input size已受官方支援。不要透過降低3D原始點解析度來
省GPU，因為nearest surface與最終geometry仍需高解析depth/cloud。

### Clustering怎麼調

先在RViz看self-filtered與tracked cloud。噪聲被拆太碎時，小幅增加
`clustering.tolerance`或降低`min_points`；相鄰物體被合併時反向調整。再用
`min_dimension`/`max_dimension`排除不合理尺寸。`pointcloud.voxel_size`應小於
希望分辨的最小間隙，workspace則只涵蓋真正robot工作區。所有變更都要重跑
crossing、sparse與突然伸手案例。

### Re-prompt怎麼調、如何看mask漂移

`fusion.minimum_mask_cluster_iou`越高越容易re-prompt；
`maximum_mask_area_change_ratio`越低越敏感；
`minimum_valid_depth_ratio`與`maximum_centroid_difference_m`分別控制depth與3D
一致性。debug image顯示`RP:<reason>`時代表該prompt由quality gate觸發。
diagnostics應同時檢查Edge latency/error；若mask覆蓋離開projection、有效depth
急降、面積跳變或3D centroid離開Kalman prediction，mask會變成
`DEGRADED/INVALID`，INVALID不會替換點雲。

### 如何確認Track ID穩定

錄製`/edgetam_tracker/obstacles`，按真實物體檢查`track_id`、state、hit/missed
與position時間序列；同時故意交會、遮擋及短暫拔RGB。正常結果是短遮擋進入
`OCCLUDED/LOST`後回到原ID；超過retention離場再出現才配置新ID。可先跑
`tools/evaluate_tracker.py`的deterministic crossing/occlusion/re-entry案例。

### 如何接既有CBF

先讓CBF在staging訂閱`/edgetam_tracker/obstacle_cloud`，確認frame、metre scale、
surface distance與topic timeout。若只能使用舊topic，停止legacy YOLO publisher
後才開`publish_legacy_alias:=true`。動態CBF可另訂閱
`TrackedObstacleArray`的velocity、三個prediction horizons與uncertainty；本
repository沒有控制端，不能替代外部CBF的STOP/release驗證。

### 實際ROS smoke結果

2026-07-31曾以現有learned-depth PointCloud2執行短時間live smoke；實際抓到
一個持續`CONFIRMED` track與非空output cloud，單次diagnostics快照為8.230 FPS、
13.576 ms total。這不是EdgeTAM/RGB-D benchmark。完整範圍、stage latency與
當場發現的out-of-order timestamp修正記錄在
[`results/ros_live_smoke_2026-07-31.md`](../results/ros_live_smoke_2026-07-31.md)。

同日另以官方checkpoint執行三次CUDA/bf16 wrapper呼叫：兩個ID的初始與
rolling-window propagation、再刪除一個ID後重建，皆產生非空mask。三次wall
clock合計751.883 ms（3.990 calls/s），wrapper latency為469.938、219.148、
61.905 ms，峰值CUDA allocated/reserved為561.634/592.000 MiB。多ID呼叫使用
上述明確標記的independent-state相容模式；這不是ROS RGB-D或控制deadline。
完整紀錄見
[`results/edgetam_official_smoke.md`](../results/edgetam_official_smoke.md)。
