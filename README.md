# Social Safety Decision — RGB-D perception, OpenArm avoidance and simulation

This repository is organized as cumulative architecture branches.  A higher
layer contains the source from the layers below it, so collaborators can test
one subsystem without losing the end-to-end integration history.

| Branch | Architecture layer | Intended use |
|---|---|---|
| `layer/perception-detection` | RGB-D, point-cloud reconstruction, YOLO/MediaPipe/EdgeTAM and tracking | Perception development and the 8080 live viewer |
| `layer/dynamic-avoidance-control` | Perception plus OpenArm MoveIt control and the dynamic safety layer | Planner/controller development without simulator assets |
| `layer/simulation-validation` | Perception and control plus Gazebo/Isaac sources, launch files and validators | Reproducible end-to-end validation |
| `main` | Reviewed integrated snapshot of `layer/simulation-validation` | New collaborators and demonstrations |
| `2d_version`, `3d_version` | Historical development lines | Read-only comparison; do not base new work here |

The system boundary follows four rules:

1. **Perception & Detection** converts synchronized RGB-D into a current-frame
   environment cloud and model-confirmed dynamic-obstacle cloud.
2. **Dynamic Avoidance & Control** consumes only ROS 2 observations, robot
   state and task goals, then sends collision-checked trajectories to OpenArm.
3. **Simulation & Validation** supplies repeatable Gazebo/Isaac scenes,
   sensors and evaluator-only ground truth; perception mode never receives
   simulator ground truth.
4. Model weights, generated USD, build/install/log directories, bags, caches
   and third-party repositories are intentionally not committed.  Their setup
   scripts and pinned source revisions are committed instead.

See [branch ownership, repository layout and current progress](docs/BRANCHES_AND_PROGRESS.md)
before changing interfaces shared by another layer.

## Quick start: perception and the 8080 viewer

```bash
git clone --branch layer/perception-detection \
  https://github.com/Laiting20021202/social-safety-decision.git
cd social-safety-decision
bash scripts/setup.sh
bash scripts/setup_edgetam.sh       # optional EdgeTAM model path
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select realtime_3d_safety_decision
bash scripts/run_koch_stream.sh
```

Open `http://<host-ip>:8080/`.  The simulator and full avoidance commands are
documented on `main` under `openarm_isaac_sim_test/README.md`.

# Interactive Temporal 4D Viewer and Safety System

The default application is a clean 4D reconstruction viewer with a focused person layer: RGB video in, temporally consistent color point clouds out, plus YOLO person masks mapped into robust 3D person centers, wireframe 3D boxes, and short direction arrows. The live viewer uses Metric Video Depth Anything Small in streaming mode, so a moving hand is estimated with temporal-attention state from earlier frames instead of an unrelated first-frame anchor. St4RTrack remains selectable. The viewer does **not** render danger volumes, ground/path planning, other object classes, or run the safety worker. The original safety pipeline remains available through the `realtime_fast`, `realtime_balanced`, `realtime_quality`, and `agx` profiles.

## Point-cloud-first EdgeTAM tracker（ROS 2）

An independent ROS 2 pipeline now tracks DBSCAN/Euclidean 3D clusters with
Kalman/Hungarian association and uses official EdgeTAM only as optional RGB
mask refinement. Its safety fallback is point-cloud-only; EdgeTAM/RGB/mask
failure never clears valid 3D obstacles. The `koch_lan` default does not
load or run the retained YOLO model until it is selected in the GUI.

The LAN dashboard at `http://<host>:8080` now exposes two real runtime modes:

- **EdgeTAM + RGB Hand Gate + 3D PointCloud** (default)
- **Legacy YOLO + RGB Hand Gate + 3D PointCloud** (retained rollback path)

The same folder shows live tracker rate, EdgeTAM latency, accepted refinement
count, and the actual point-cloud prompt/mask debug image. The YOLO checkpoint
selector is shown only for the YOLO path. A ROS-side mux is the sole publisher
of the controller-compatible obstacle-cloud topic and changes source after the
newly selected pipeline has produced a cloud. The selection is sticky: stale
input holds that selected output fail-closed and never silently jumps to the
other model; only another GUI selection changes it:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select realtime_3d_safety_decision
bash scripts/run_koch_stream.sh
```

See [the complete setup, topic/frame contract, fail-safe rules, and rollback
guide](docs/edgetam_pointcloud_tracker.md) and the
[legacy-to-new pipeline analysis](docs/edgetam_pointcloud_analysis.md). The
deterministic synthetic point-cloud report is in
[results/edgetam_pointcloud_evaluation.md](results/edgetam_pointcloud_evaluation.md);
an actual official-checkpoint CUDA/API smoke is recorded in
[results/edgetam_official_smoke.md](results/edgetam_official_smoke.md).
These are build/functional results, not live RGB-D accuracy claims. The
standalone Edge launch keeps its legacy PointCloud2 alias off. In the integrated
service, Edge and YOLO publish private candidate clouds and the mux alone owns
the historical controller topic. The GUI now renders the pinned official
OpenArm v1.0 bimanual URDF (body + both arms) and animates it directly from `/joint_states` in
the same metric work-plane view as the point cloud and obstacle volumes. Robot
URDF self-filtering is still off until the camera-to-base extrinsic and link
collision coverage are physically verified; `obstacle_cloud` may therefore
still contain robot points.

Install/generate the OpenArm description once, then start the integrated
service normally:

```bash
bash scripts/setup_openarm.sh
systemctl --user restart realtime-safety-3d.service
```

The expected controller names are `openarm_left_joint1` through
`openarm_left_joint7` and `openarm_right_joint1` through
`openarm_right_joint7`; side-preserving aliases are accepted by the GUI.
If no JointState has arrived, the arm remains visible in zero pose and the
OpenArm panel reports `WAITING` instead of presenting simulated motion as live.
The detected 8 × 8 cm AprilTag sets point-cloud scale and anchors the body at
the configured `[0, -0.50, 0] m` offset. Use **OpenArm 基座外參 / Base
calibration** only to refine that measured Tag→Body transform. See the [OpenArm GUI and coordinate
contract](docs/openarm_gui.md).

The integrated service now feeds Edge 3D tracks through the same versioned
`/realtime_safety/arm_obstacle_relationships` JSON publisher used by YOLO. It
contains the persistent track ID, velocity/motion state, obstacle center, Koch
arm center, center distance and surface clearance. This repository still does
not contain the external CBF/FK/Jacobian/motor controller, so motor-side
integration must retain its timeout and STOP checks.

The asynchronous path publishes point-cloud safety first. A later EdgeTAM
result is fused only with the exact saved RGB/geometry context that created its
sequence, and a same-measurement-stamp correction is allowed only if the
context is still within the stale timeout and no newer safety output has been
published. Old masks are discarded rather than applied to current geometry, so
enabling EdgeTAM does not mean every output is refined or zero-latency.
If a previously segmented track becomes `OCCLUDED` while current raw depth is
still present, a valid exact-context mask may add spatial/depth-gated points;
this remains an INVALID-quality prediction with capped confidence and growing
uncertainty, not a fresh 3D measurement or a new detection. Complete geometry
loss still has only bounded Kalman prediction.
The pinned upstream source needed one PyTorch 2.9 compatibility correction:
an expanded zero-stride latent tensor must use `reshape()` rather than
`view()`. With that patch, two objects stay in one grouped predictor state
instead of running one full inference per ID. The production one-frame
latest-only setting measured 44.8–52.3 ms after first-call CUDA warm-up on the
RTX 4060 Ti (roughly 19–22 model calls/s, above the 12 Hz pipeline target).
The live `koch_lan` run synchronizes RGB, generated CameraInfo, and
learned-depth PointCloud2. The fixed-camera profile now calibrates a short startup 3D
background baseline, removes the wooden panel and stationary fixtures before
clustering, rejects near-full-frame prompts, and reports prompt/mask gate counts
in the UI. Keep the workspace empty until the UI says `Foreground filter:
ready`. A live pre-baseline foreground candidate produced accepted DEGRADED
masks and 34 exact-context refined corrections, proving the mask/cloud
correction path now executes; that candidate was a stationary lower-edge
fixture, not a labelled hand, so hand recall and accuracy are still unproven.

There is no synchronized RGB-D rosbag or ground truth in this repository.
Consequently, synchronized EdgeTAM accuracy, configured self-filter recall,
and external CBF/hardware metrics remain N/A. The live CameraInfo is generated
from the Koch profile's provisional `fx=fy=272`, `cx=159.5`, `cy=119.5`
projection because the remote camera publishes an empty frame and all-zero
K/P. `CameraInfo` is a
latest-only subscription rather than an approximate-synchronizer member; the
node separately validates its shape/frame and requires either a matching stamp
or explicitly allowed zero-stamp static calibration. During a short geometry
gap, only bounded predictions of already observed tracks may be published;
they cannot create new obstacles, are marked `prediction_only`, and stop after
the configured stale timeout.

## One-command viewer

```bash
cd /path/to/realtime_3d_safety_decision
bash scripts/run_viewer.sh
```

Open <http://127.0.0.1:8080>, upload a video, and the viewer will populate `/frames/t...` with bounded 4D reconstruction history. Left drag rotates, the wheel zooms, and right drag pans. To open a video directly:

```bash
bash scripts/run_viewer.sh /path/to/video.mp4
```

The pipeline is designed around one safety rule: stale frames are less useful than current frames. Every inter-worker queue is bounded (default size 2), drops its oldest item under pressure, and records the drop. Models run outside the GUI renderer, safety prediction continues at its own target rate, and no complete-video preprocessing or per-frame disk round-trip is used.

## What is implemented

- MP4, AVI, MOV, MKV, WEBM, webcam, and RTSP sources, including pause, resume, restart, loop, seek, and playback speed.
- GUI upload and command-line source selection.
- The Koch reconstruction viewer uses YOLO26m-seg restricted to people. Its multi-scale mask prototype improves the 2D boundary used to select obstacle points, while ByteTrack recovers established IDs through blur or partial occlusion.
- Person-mask boundaries and background depth layers are rejected before estimating the robust 3D center and tight percentile box. Predicted holds never create a new depth measurement.
- Confirmed GUI boxes survive six missed reconstruction updates and are explicitly labelled `HOLD`. The ROS obstacle cloud uses a separate conservative 24-update hold plus motion alignment, so isolated segmentation/depth failures do not flash an obstacle off.
- Walking direction uses recent 3D-center displacement with smoothed Kalman velocity as a sparse-update fallback. ID jumps are re-anchored before they can create a large false arrow.
- The RTX 4060 Ti viewer profile uses a measured 280px streaming video-depth input, independent CUDA streams, 24 FPS 640px sidebar video delivery, and sends each 3D frame only once instead of retransmitting it on every video frame. A 640×480 USB webcam measured 21.6 display FPS and 11.7 3D FPS in an end-to-end 120-frame run.
- Live playback uses one persistent WebGL point-cloud/people node set and applies aligned updates atomically. The 64-frame scene history is disabled in this profile because allocating and toggling frame nodes caused visible white flashes in browsers. Presentation mode also pins the WebGL device-pixel ratio on iPadOS and uses a dark canvas backing surface so Safari cannot expose a white frame while reallocating its drawing buffer.
- The optional St4RTrack backend retains outdoor filtering for top-connected blue/bright low-texture sky and implausibly near upper-image geometry. The default temporal-depth backend uses a 24k hard point budget while retaining its aligned dense pointmap for person-center and 3D-box extraction.
- Persistent YOLO11n-seg CUDA backend in the optional safety profiles; COCO obstacle classes are normalized to person, bicycle, chair, bag, suitcase, vehicle, and related obstacle types.
- Timestamp-aware stable 2D IDs, short-occlusion association, and between-segmentation prediction.
- Real Depth Anything V2 Small relative monocular depth point maps; no synthetic point cloud fallback.
- Real Metric Video Depth Anything Small streaming inference with a persistent multi-frame temporal-attention cache; the cache is reset when the input source changes or restarts.
- In-memory St4RTrack frame-pair adapter. It directly returns tracking/reconstruction point maps and never writes NPY files.
- Light-theme 4D viewer with centered color point clouds, automatic initial framing, current-frame playback, optional history display, and a bounded `/frames` scene tree.
- Mask-to-pointmap 3D clusters plus radius-connected `unknown_obstacle` clustering.
- Constant-velocity 3D Kalman filters with timestamp-derived velocity and static/dynamic hysteresis.
- Three-second trajectory prediction and an uncertainty-inflated series-of-ellipsoids swept volume.
- RANSAC ground plane, traversable-region filtering, nine local path candidates, collision rejection, detour selection, and all-paths-blocked STOP.
- SAFE, CAUTION, WARNING, STOP, and DEGRADED with immediate STOP and debounced release.
- Persistent Viser scene handles for point clouds, bounding boxes, arrows, trails, future paths, red/orange danger volumes, green traversable ground, candidates, and the selected cyan path. Updating a frame never writes client camera state, so the viewing angle is preserved.
- Actual input/display/segmentation/3D/safety rates, avg/p95 latency, drops, queue usage, RAM, and allocated VRAM.
- Streaming JSONL, trajectory CSV, optional annotated MP4 and final binary PLY.
- A benchmark report with measured rates and RAM/VRAM trend estimates.
- Direct ROS 2 `sensor_msgs/PointCloud2` publication plus the optional JSONL safety bridge.

## Coordinate and scale contract

The internal robot-oriented coordinate convention is:

- `x`: right
- `y`: forward
- `z`: up

The default `relative` mode intentionally does not claim metres or m/s. JSONL marks `metric_valid=false`, records `velocity_unit=relative_units/s`, and the GUI shows `RELATIVE SCALE`. Relative-space trajectory intersection and risk remain available.

Metric output is enabled only for aligned RGB-D input or calibrated monocular input with an explicit positive `--manual-scale`. Camera-height-only automatic scale recovery is not claimed as verified metric output.

## Architecture

```text
Capture worker ─┬─> latest-frame queue(2) ─> Fast perception worker ─┐
                └─> latest-frame queue(2) ─> 3D worker ──────────────┤
                                                                    v
                  Safety worker at target 10 Hz <─ latest bounded state
                                      │
                                      ├─> streaming JSONL / CSV
                                      v
                     GUI renderer ─> persistent Viser scene + video
```

Full 3D reconstruction is allowed to run below 10 Hz. Every safety tick predicts the latest 3D states forward using the Kalman state, regenerates danger volumes and local paths, and publishes a current decision. A later point map corrects the filters. In `hybrid`, St4RTrack performs lower-rate world correction while Depth Anything supplies current relative point maps between corrections. If St4RTrack is absent, the application records the fallback and continues with real monocular depth.

## Install

Python 3.10–3.12 and an NVIDIA CUDA setup supported by PyTorch are recommended.

```bash
git clone --branch 3d_version https://github.com/Laiting20021202/social-safety-decision.git realtime_3d_safety_decision
cd realtime_3d_safety_decision
bash scripts/setup.sh
source .venv/bin/activate
bash scripts/download_models.sh --viewer
```

Or install into an existing environment:

```bash
python3 -m pip install -r requirements.txt
bash scripts/download_models.sh --viewer
```

TensorRT is optional and never required to start:

```bash
python3 -m pip install -r requirements_tensorrt.txt
```

The AGX profile expects an exported `yolo11n-seg.engine`. Export it on the target architecture with Ultralytics before selecting that profile.

## Viewer depth models

```bash
bash scripts/download_models.sh --viewer
```

This shallow-clones the official Video Depth Anything and St4RTrack repositories into ignored `third_party/` folders. It caches the Apache-2.0 Metric Video Depth Anything Small checkpoint and the optional St4RTrack sequence checkpoint. The external St4RTrack code/model has a non-commercial scientific research license; review it before use.

The default `video_depth` backend consumes one live frame at a time but keeps the official streaming model's temporal cache. If its external code or checkpoint is unavailable, the application reports the cause and falls back to per-frame Depth Anything V2. Select the old backend explicitly with:

```bash
bash scripts/run_viewer.sh --depth-mode st4rtrack
```

The adapter contract is:

```python
adapter.load()
adapter.warmup()
adapter.set_anchor(frame_packet)
pointcloud = adapter.infer(anchor_packet, current_packet)
adapter.reset()
adapter.close()
```

Both inputs are memory-resident arrays/tensors. `pred2["pts3d_in_other_view"]` supplies current reconstruction and `pred1["pts3d"]` supplies tracked anchor content, converted into x-right/y-forward/z-up coordinates, confidence-filtered, and immediately voxel-downsampled.

### MASt3R-SLAM dense point clouds

Install the official CVPR 2025 MASt3R-SLAM code, CUDA extension, and three
upstream checkpoints in an isolated environment:

```bash
bash scripts/setup_mast3r_slam.sh
```

The setup needs an NVIDIA CUDA toolkit with `nvcc` and at least 8 GiB free
disk space. The separate `.venv-mast3r-slam` is intentional: MASt3R-SLAM and
St4RTrack ship incompatible top-level `dust3r` modules. The application sends
RGB arrays to that process in memory and receives the globally aligned current
pointmap plus a bounded, voxel-downsampled accumulated map; it does not write
frames to disk.

Start normally, then use **點雲生成方法 / Point Cloud Method** in the GUI to
switch among Metric Video Depth Anything, MASt3R-SLAM, St4RTrack, and per-frame
Depth Anything. RGB preview remains live while a method loads. A missing or
failed method is reported in the same control and falls back to a working
method. MASt3R-SLAM can also be selected at launch:

```bash
bash scripts/run_viewer.sh --depth-mode mast3r_slam
```

MASt3R and MASt3R-SLAM code/checkpoints have upstream research and
non-commercial license terms; review their `LICENSE.md` and
`thirdparty/mast3r/CHECKPOINTS_NOTICE` before use.

## Run

Default 4D viewer with YOLO 3D person boxes/centers/arrows, but no danger zones:

```bash
bash scripts/run_viewer.sh
# equivalent:
.venv/bin/python app.py --profile st4rtrack_viewer --device cuda
```

The right sidebar lists every current YOLO person with ID and confidence. `pending` means it has appeared once and is visible for inspection but is not yet allowed to create a 3D box; `confirmed` means it passed the consecutive-frame gate.

Disable the person layer when an entirely bare point cloud is preferred:

```bash
bash scripts/run_viewer.sh --no-people
```

Optional safety profile:

```bash
python app.py --source /path/to/test.webm --profile realtime_fast --device cuda
# or
bash scripts/run_fast.sh /path/to/test.webm
```

USB webcam is detected automatically when `--source` is omitted. The detector
opens the available camera nodes, rejects metadata-only nodes, and selects the
first device that produces a real frame. Webcam RGB frames feed the same live
temporal-depth reconstruction path as uploaded videos:

```bash
bash scripts/run_viewer.sh
# explicit auto-detection:
python app.py --source auto --profile st4rtrack_viewer --device cuda
# explicit device index or Linux device path:
python app.py --source 0 --profile st4rtrack_viewer --device cuda
python app.py --source /dev/video0 --profile st4rtrack_viewer --device cuda
```

Use `--no-auto-webcam` to keep the GUI idle until a video is uploaded. In the
GUI, press **Auto-detect USB webcam** to rescan after connecting or reconnecting
a camera.

If the webcam calibration matrix is `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
at the capture resolution, pass it directly to the point-cloud projection:

```bash
bash scripts/run_viewer.sh --focal-x 615.2 --focal-y 614.7 --principal-x 319.4 --principal-y 241.1
```

Accurate intrinsics improve the 3D ray directions (`x` and `z`), especially near
the image boundary. They cannot by themselves determine a hand's forward depth
from one RGB camera; the streaming video-depth model supplies the multi-frame
prior used for that ambiguity. Although this backend uses metric weights, keep
safety output in relative mode until its scale has been checked against a
measured distance for this webcam and environment.

For a rigid camera/reference setup, a known distance can calibrate the remaining
global monocular scale. The ROI is normalized
`[x_min, y_min, x_max, y_max]` and must cover the fixed foreground reference:

```bash
python app.py \
  --source 0 \
  --profile st4rtrack_viewer \
  --depth-reference-m 0.40 \
  --depth-reference-roi 0.46 0.34 0.61 0.70
```

The calibrator uses the configured foreground percentile over eight frames,
then follows slow model drift while rejecting abrupt changes caused by a hand
or person crossing the ROI. The scale is applied before projection and voxel
sampling, so the GUI, ROS PointCloud2, tracking, and avoidance calculations use
the same calibrated coordinates. Raw model outliers are rejected before
scaling, and `max_metric_depth_m` can bound the calibrated operational range.
One reference distance calibrates scale only; accurate `fx/fy/cx/cy` still
requires a checkerboard calibration.

## Koch camera input and ROS 2 LAN outputs

The repository includes a one-command LAN configuration. It consumes the
camera's ROS-backed MJPEG endpoint, keeps the Viser GUI on the LAN, republishes
a host-local ROS 2 preview for RQT, and publishes every latest reconstruction
as a ROS 2 PointCloud2:

```bash
source /opt/ros/humble/setup.bash
bash scripts/run_koch_stream.sh
```

Defaults:

- Network input: the camera's `web_video_server` MJPEG stream backed by
  configurable `CAMERA_INPUT_TOPIC` (default `/rgbd/color/image_raw`), scaled to `320x240`. Sending compressed JPEG
  across Wi-Fi avoids the DDS fragmentation/ACK stall observed with 153–230 KB
  raw frames.
- Local RQT preview: `/realtime_safety/camera/image_raw` (`bgr8`, maximum
  `10 Hz`). Use `bash scripts/run_rqt_camera.sh`; it sources the correct domain
  and avoids opening a second high-bandwidth subscription across Wi-Fi.
- Paired calibration: `/realtime_safety/camera/camera_info`, using the same
  acquisition stamp and `realtime_safety_frame` as the preview. Its pinhole
  intrinsics come from the provisional Koch profile values below.
- Point cloud: `/realtime_safety/pointcloud` (`sensor_msgs/msg/PointCloud2`)
- Controller-compatible obstacle cloud:
  `/realtime_safety/yolo_obstacles/pointcloud`
  (`sensor_msgs/msg/PointCloud2`). The historical topic name is retained only
  for controller compatibility; its sole publisher is the Edge/YOLO source
  mux.
- Native EdgeTAM outputs: `/edgetam_tracker/obstacles`,
  `/edgetam_tracker/obstacle_cloud`, `/edgetam_tracker/debug_image`, and
  `/edgetam_tracker/diagnostics`.
- The left 3D viewer subscribes to `/edgetam_tracker/obstacle_cloud` and draws
  extracted foreground obstacles as larger bright points. A width-zero safety
  cloud clears the overlay immediately instead of leaving a stale obstacle.
- Runtime selector service: `/edgetam_tracker/set_enabled`
  (`std_srvs/srv/SetBool`). Disabling it stops visual refinement but leaves
  point-cloud clustering/tracking and obstacle output active.
- Arm/obstacle relationship:
  `/realtime_safety/arm_obstacle_relationships` (`std_msgs/msg/String`). Both
  backends use schema version 1; Edge tracks provide the same center, velocity,
  motion state and distance fields as the retained YOLO path.
- Point-cloud rate: maximum `12 Hz`, with `8,000` latest points
- Frame ID: `realtime_safety_frame`
- Metric reference: the detected AprilTag black square is `0.08 m` per side;
  four reconstructed sides and two diagonals constrain global scale, shown as
  `APRILTAG … METRIC LOCK` in System Overview
- Camera geometry: because the remote webcam publishes an all-zero
  `CameraInfo.K`, the Koch app publishes a local provisional calibration using
  the original pre-calibration
  `60.93° × 47.61°` projection (`fx=fy=272 px` at `320×240`). Open
  **Camera Geometry** in the GUI to tune horizontal/vertical FOV live while
  inspecting RViz. Changing focal geometry requires the AprilTag metric lock to
  be reacquired before using distances.
- Angled camera correction: **Camera Geometry / 俯視梯形修正** has
  downward-pitch, roll, and yaw controls plus **套用俯視梯形 / Top-down
  trapezoid view**. The correction rotates
  reconstruction, tracks, arm geometry, and EdgeTAM foreground together. It is
  display-only, so RGB projection and controller-facing ROS coordinates do not
  change. A rear/upper mount therefore produces the expected perspective
  trapezoid when viewed from above without corrupting safety geometry.
- ROS domain: `42`
- DDS: Cyclone DDS, with LAN-wide multicast discovery plus `192.168.0.231` as
  an explicit unicast fallback peer
- GUI: `http://<this-computer-LAN-IP>:8080`
- GUI layout: recording-friendly presentation mode with an EdgeTAM/YOLO
  selector, real diagnostics, and a live prompt/mask debug preview
- Sidebar: drag its cyan left edge to resize it; the width is restored after a
  refresh. Double-click the edge to reset it, or use the top-right arrow to
  collapse the panel completely.

The network capture and ROS publishers always retain only the newest frame. A
camera outage does not stop the GUI process; capture reconnects automatically.
On every launch, the host asks `/koach_webcam` to use YUV422 before the camera's
local `web_video_server` compresses it. The PointCloud2 publisher uses
reliable, volatile, keep-last-1 QoS. This works with RViz's default reliable
subscription and remains compatible with the camera computer's best-effort
`koch_vamp_pointcloud_server` reader. Publishing still runs outside the
reconstruction worker, so a slow network subscriber cannot stall inference.
It publishes `x`, `y`, `z` float fields and packed RGB, using this repository's
`x-right, y-forward, z-up` coordinate convention.

The `koch_lan` profile uses `320px / 20 FPS` GUI video and targets
`12 Hz / 8,000 points` for reconstruction and ROS point clouds.
PointCloud2 serialization runs in latest-only publisher workers, so a DDS
subscriber cannot block depth or EdgeTAM inference. RGB video also has its own
renderer, so point-cloud serialization cannot stall it.
Use `--no-presentation-mode` when you want the original analysis-first layout;
other profiles can opt into the recording layout with `--presentation-mode`.

For geometry tuning, increase a FOV slider when that point-cloud axis is too
narrow, or decrease it when the axis looks stretched. Use the same physical
object near the image center and edges while checking the result. The GUI
change is immediate but intentionally session-local. For a rear/upper camera,
first increase **Camera downward tilt** until the work surface is level, correct
sideways lean with **roll**, then set its heading with **yaw** and press the
top-down button. After choosing verified values, copy the angles and displayed
`fx/fy` into `configs/koch_lan.yaml`. A printed checkerboard calibration remains
the correct way to replace these provisional values and estimate lens
distortion.

The Koch reference scale is computed robustly over the eight-frame warmup and
then frozen because the camera/reference mount is rigid. This avoids global
scene-size drift when a hand crosses the reference ROI. Restarting the source
or service deliberately performs a fresh warmup.

On another ROS 2 Humble computer in the same network:

```bash
source /opt/ros/humble/setup.bash
export ROS_LAN_PEER=192.168.0.234
source /path/to/realtime_3d_safety_decision/scripts/ros2_lan_env.sh
ros2 daemon stop
ros2 topic list --no-daemon --spin-time 8
ros2 topic info /realtime_safety/pointcloud --verbose --no-daemon
ros2 topic hz /realtime_safety/pointcloud
ros2 topic echo /realtime_safety/pointcloud --once --qos-profile sensor_data --no-daemon
ros2 topic info /realtime_safety/yolo_obstacles/pointcloud --verbose --no-daemon
ros2 topic echo /realtime_safety/yolo_obstacles/pointcloud --once --no-daemon
```

To make that client setup persistent, copy `scripts/ros2_lan_env.sh` to
`~/.config/ros2/lan_env.sh` and add these lines to its `~/.bashrc`:

```bash
export ROS_LAN_PEER=192.168.0.234
source "$HOME/.config/ros2/lan_env.sh"
```

For RViz2, run `bash scripts/run_rviz_pointcloud.sh`. The included configuration
selects `/realtime_safety/pointcloud`, uses **Reliable** QoS, renders the packed
RGB field, and sets `realtime_safety_frame` as the fixed frame. DDS discovery
requires the same `ROS_DOMAIN_ID`,
`ROS_LOCALHOST_ONLY=0`, and multicast/ROS 2 UDP traffic allowed by the firewall.

Override any network default without editing code:

```bash
KOCH_CAMERA_SOURCE='ros2:///camera/image_raw' \
POINTCLOUD_TOPIC=/my_robot/pointcloud \
CAMERA_INFO_TOPIC=/my_robot/camera_info \
EDGETAM_CONFIG=/path/to/edgetam_pointcloud_tracker.yaml \
KOCH_ROS_DOMAIN_ID=7 \
bash scripts/run_koch_stream.sh
```

Set `CAMERA_SOURCE=ros2:///my_camera/image_raw` to test another direct
DDS raw input. The default is the compressed MJPEG bridge because this Wi-Fi
path stalled reliable raw DDS after the first sample.

The launch script selects the LAN interface used to reach `ROS_LAN_PEER`, uses
SPDP multicast for LAN-wide discovery, and retains direct unicast discovery to
the configured peer. Participant indices are selected with `auto`, so the
publisher, `ros2` CLI, RViz2, and other nodes can coexist on the same computer
without contending for participant ports. User data remains unicast.
Override the peer or interface when the addresses change:

```bash
ROS_LAN_PEER=192.168.0.231 ROS_LAN_INTERFACE=wlan0 bash scripts/run_koch_stream.sh
```

If RGB and point clouds pause together, first measure the camera path with
`ping 192.168.0.231`. Multi-second latency is a Wi-Fi problem rather than a
depth/GUI problem. Disable Wi-Fi power saving persistently on **both** computers
(this briefly reconnects the selected Wi-Fi connection):

```bash
PEER_IP=192.168.0.234  # on the publisher (.234), use 192.168.0.231 instead
IF=$(ip route get "$PEER_IP" | sed -n 's/.* dev \([^ ]*\).*/\1/p')
CON=$(nmcli -g GENERAL.CONNECTION device show "$IF")
nmcli connection modify "$CON" 802-11-wireless.powersave 2
nmcli connection down "$CON" && nmcli connection up "$CON"
```

For the Koch NUC, the repository also includes a persistent network fix. It
disables NetworkManager Wi-Fi power saving, Realtek `rtw88` deep power saving,
and USB autosuspend for the Edimax adapter, then reapplies those settings at
every boot:

```bash
# Run this on 192.168.0.231. It asks for sudo once.
bash scripts/install_koch_wifi_stability.sh
```

Its system service is `koch-wifi-stability.service`; inspect it with
`sudo systemctl status koch-wifi-stability --no-pager`.

If ping works but the server accepts TCP and returns no bytes, the remote
`web_video_server` is hung. Confirm from this computer:

```bash
curl --max-time 5 -o /dev/null -w '%{http_code} %{size_download}\n' \
  'http://192.168.0.231:8080/snapshot?topic=/rgbd/color/image_raw'
```

An HTTP code `000` with `0` bytes requires restarting `web_video_server` on the
NUC. Use its existing systemd/ROS launch service when available. If it was
started manually, it can be recovered with:

```bash
pkill -TERM -f '[w]eb_video_server'
nohup ros2 run web_video_server web_video_server --ros-args \
  -p port:=8080 -p address:=0.0.0.0 -p server_threads:=4 -p ros_threads:=2 \
  >"$HOME/.ros/web_video_server.log" 2>&1 &
```

If a receiving computer does not have Cyclone DDS installed, install
`ros-humble-rmw-cyclonedds-cpp` or start both computers with another RMW that
is installed on both sides. Stop the ROS daemon after changing RMW or domain
settings so that it does not retain the old environment.

Hybrid:

```bash
python app.py --source /path/to/test.mp4 --profile realtime_fast --depth-mode hybrid --device cuda
```

AGX:

```bash
python app.py --source /path/to/test.mp4 --profile agx --device cuda
```

Safety GUI upload only:

```bash
python app.py --profile realtime_fast --device cuda
```

Open the printed Viser URL (normally <http://127.0.0.1:8080>). Left drag rotates, the wheel zooms, and right drag pans.

Useful output flags:

```bash
python app.py --source test.mp4 --record --export-ply --output-dir sessions/demo
```

Safety JSONL and trajectory CSV are enabled by default. Use `--no-log` to disable them.

## Profiles

| Profile | Mode | 3D input | Maximum displayed points | Intended use |
|---|---:|---:|---:|---|
| `st4rtrack_viewer` | Reconstruction + YOLO11s people at 640 | Streaming metric video depth 280 (St4RTrack optional) | 24,000 | RTX 4060 Ti live viewer with held person boxes and walking arrows |
| `koch_lan` | LAN point-cloud tracking + EdgeTAM refinement | 320x240 MJPEG + streaming metric video depth 252 | 8,000 | EdgeTAM UI/debug plus fail-safe controller cloud; no YOLO in the default runtime |
| `realtime_fast` | Safety + YOLO 320 | Depth/St4 224 | 30,000 | Latency-first safety mode |
| `realtime_balanced` | Safety + YOLO 416 | Depth/St4 320 | 60,000 | More 3D detail |
| `realtime_quality` | Safety + YOLO 640 | St4RTrack 512 | 100,000 | Safety visualization quality |
| `agx` | Safety + TensorRT YOLO | Depth/St4 224 | 20,000 | Serialized low-memory Jetson mode |

The adaptive controller watches p95 latency, display rate, queue pressure, and VRAM. Under sustained pressure it lowers point count, reconstruction input/rate, and segmentation quality while leaving `safety.target_hz` unchanged. The GUI shows `DEGRADED`; recovery is gradual after a stable interval.

## Benchmark

```bash
python scripts/benchmark.py \
  --source /path/to/test.mp4 \
  --profile realtime_fast \
  --depth-mode fast_depth \
  --duration 60 \
  --output outputs/benchmark_report.json
```

Rates are event-derived and are never hard-coded. Latency measures current frame capture to safety publication. A dropped count can exceed the number of source frames because one stale frame may be dropped independently from the fast and 3D branches.

### Verified five-minute result

Measured on 2026-07-13 with an NVIDIA GeForce RTX 4060 Ti 16 GB, PyTorch 2.9.0/CUDA 12.8, Python 3.10.12, `realtime_fast`, `fast_depth`, and the official St4RTrack `assets/feng.mp4` looped for 300.04 seconds. This is not the user's separate reference video.

| Metric | Actual |
|---|---:|
| Input | 30.00 FPS |
| Video display | 29.28 FPS |
| YOLO11n segmentation | 8.27 FPS |
| Depth/3D update | 7.14 FPS |
| Safety update | 9.996 FPS |
| Average latency | 45.38 ms |
| p95 latency | 98.07 ms |
| Dropped branch frames | 467 |
| Queue at end / capacity | 0 / 2 |

After the first 20% warm-up/load interval, RAM start-to-end changed by -0.29 MB and allocated VRAM by -0.18 MB. Linear slopes were +0.41 MB/min and +0.26 MB/min respectively, within allocator/sampling fluctuation and with no sustained start-to-end growth. The complete machine-readable report is `benchmarks/benchmark_5min_rtx4060ti.json`.

The stable segmentation and 3D rates did **not** reach 10 Hz on this serialized GPU schedule; the measured safety loop did. The system reports these actual rates and does not relabel them as 10 Hz.

## Test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

The environment variable below enables the real neural-model/CUDA smoke test; otherwise it is explicitly skipped:

```bash
SAFETY_SMOKE_VIDEO=/path/to/short.mp4 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_pipeline_smoke.py -q
```

Coverage includes queue bounds/drop-oldest behavior, real timestamps, stable IDs, mask-to-3D, point count bounds, 3D velocity, directional danger zones, no long stationary sweep, STOP hysteresis, relative-scale labeling, RANSAC ground, local detour/STOP, camera motion, asynchronous renderer behavior, bounded Viser nodes, JSONL, and PLY.

## Outputs

```text
sessions/<timestamp>/
├── safety.jsonl
├── trajectories.csv
├── annotated.mp4       # with --record
└── final_pointcloud.ply # with --export-ply
```

ROS 2 instructions are in `realtime_safety/ros2_bridge/README.md`.

## Known limits

- A single monocular RGB stream remains ambiguous at occlusion boundaries. The streaming metric model substantially improves temporal consistency but is not a substitute for stereo or RGB-D sensing in a certified safety function.
- The Video Depth Anything streaming implementation is marked experimental by its authors and has lower benchmark accuracy than their offline 32-frame inference. This viewer chooses it because offline inference cannot provide live output.
- The provided St4RTrack adapter is pairwise feed-forward. Upstream test-time adaptation is intentionally excluded from the real-time path.
- YOLO11n-seg uses COCO classes and cannot natively distinguish every requested fine-grained class (for example wheelchair versus chair); unmatched geometric clusters are `unknown_obstacle`.
- Unknown 3D clustering is intentionally low-rate and conservative to protect safety latency.
- Camera motion is estimated as robust 2D affine motion, not a full metric 6-DoF pose.
- The local planner is a `DEMO FORWARD CORRIDOR`, not Nav2 or a replacement for a certified robot safety controller.
- TensorRT, RGB-D camera capture, actual AGX hardware, and the official St4RTrack checkpoint require target-specific assets/hardware and are not silently reported as tested.
- Viser supports the requested interaction and persistent view, but uploaded videos are copied to a bounded per-process temporary file and removed by the operating system's temp cleanup policy.

## References

- [Official St4RTrack repository](https://github.com/HavenFeng/St4RTrack)
- [Official MASt3R-SLAM repository](https://github.com/rmurai0610/MASt3R-SLAM)
- [MASt3R-SLAM project page](https://edexheim.github.io/mast3r-slam/)
- [MASt3R-SLAM CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html)
- [St4RTrack project page](https://st4rtrack.github.io/)
- [St4RTrack paper](https://arxiv.org/abs/2504.13152)
- [Official Video Depth Anything repository](https://github.com/DepthAnything/Video-Depth-Anything)
- [Video Depth Anything paper](https://arxiv.org/abs/2501.12375)
- [OpenCV camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Viser](https://viser.studio/)
- [Official OpenArm documentation](https://docs.openarm.dev/api-reference/description/)
- [Official OpenArm description repository](https://github.com/enactic/openarm_description)
