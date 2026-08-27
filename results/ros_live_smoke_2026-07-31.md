# ROS 2 live point-cloud smoke test — 2026-07-31

This record covers one short run against the repository's already-running
`/realtime_safety/pointcloud` publisher. It is not an RGB-D, EdgeTAM, robot, or
CBF benchmark.

## Scope

- Built the package into a clean `/tmp` colcon build/install tree.
- Started `edgetam_pointcloud_tracker_node` with EdgeTAM, debug publishers, and
  the legacy controller alias disabled.
- Used the existing learned-depth PointCloud2 plus RGB topics.
- Read one diagnostics message, one `TrackedObstacleArray`, and one output
  PointCloud2 message through DDS.
- Stopped the node with SIGINT after inspection.

## Observed output

| Item | Observed value |
|---|---:|
| Pipeline FPS snapshot | 8.230 |
| Total latency snapshot | 13.576 ms |
| Preprocessing | 1.288 ms |
| Clustering | 9.583 ms |
| 3D tracking | 0.361 ms |
| Fusion adapter | 0.052 ms |
| Input / processed points | 2,208 / 353 |
| Clusters / active tracks | 1 / 1 |
| Output obstacle-cloud width | 2,139 points |
| Output frame | `realtime_safety_frame` |
| Track state / ID | `CONFIRMED` / `1` |
| Point-cloud / mask quality | `GOOD` / `UNAVAILABLE` |
| Nearest surface distance | 0.33355 m |
| Prediction samples | 0.2, 0.5, and 1.0 s (3 points) |

The output cloud and obstacle message shared the input measurement stamp and
the configured tracking frame. The obstacle message contained OBB pose/size,
velocity, nearest surface sample, lifecycle counters, confidence, and three
future positions.

## Failure found and corrected

Approximate synchronization plus delayed geometry fallback produced an older
bundle 0.083408 s after a newer synchronized bundle. The first adapter version
treated every backward timestamp as a stream reset, which could churn track
IDs. The node now:

- drops a small out-of-order regression as a stale measurement;
- preserves the current tracker state and reports WARN diagnostics; and
- reserves a temporal-state reset for a regression larger than the bounded
  clock-reset threshold.

The corrected run remained alive for the test duration and continued publishing
Track ID 1. An additional bounded prediction-only path publishes existing
Kalman tracks during a short geometry gap, marks those messages explicitly, and
stops all safety output after the configured stale timeout.

## Isolated ROS graph fail-safe smoke

After the live observation, `tools/ros_tracker_integration_smoke.py` was run
against the final built messages/node. It published five synthetic 120-point
clouds on isolated topics, then stopped the geometry source. The executable
verified all of the following and exited zero:

- the same track reached `CONFIRMED`;
- output PointCloud2 remained nonempty with 120 points;
- exactly three configured prediction horizons were present;
- a short gap produced the same Track ID with `prediction_only=true` and
  `OCCLUDED`/`LOST` state;
- prediction Header time was newer than the retained
  `last_measurement_stamp`; and
- after the 0.60 s stale timeout, pipeline diagnostics became ERROR and no
  further obstacle messages were published.

Actual result:

```text
PASS ROS tracker smoke: track_id=1, measurement_points=120,
output_cloud_points=120, prediction_only=true, stale_output_stopped=true
```

## Not measured

EdgeTAM was intentionally disabled during this ROS point-cloud run; source and
checkpoint installation was completed afterward. A separate actual
official-checkpoint CUDA/API test is recorded in
`results/edgetam_official_smoke.md`, but it is not a synchronized ROS RGB-D
benchmark. There was no metric RGB-D bag, valid CameraInfo/TF calibration,
URDF self-filter geometry, external CBF, or motor hardware. Therefore ROS
EdgeTAM end-to-end performance, mask/fusion accuracy, self-filter recall, and
controller safety remain N/A.

## 8080 EdgeTAM integration follow-up

Later on 2026-07-31 the live `realtime-safety-3d.service` was changed to start
the reconstruction app and EdgeTAM tracker together. The app no longer passed
the legacy YOLO obstacle publisher or arm-relationship publisher arguments,
and `koch_lan.people_overlay=false`; no YOLO worker/model appeared in the new
service log.

Observed live contract:

| Item | Observed value |
|---|---:|
| UI | EdgeTAM / PointCloud Only dropdown, live prompt/mask image |
| Edge device / precision | CUDA / bf16 |
| Tracker snapshots | 8.294–8.345 Hz |
| Edge latency snapshots | 59.647–104.416 ms |
| Projection error | empty |
| Synchronized bundle | `True` |
| Cluster / track count | 1 / 1 |
| Debug image | publisher and live UI subscriber matched |
| Controller-compatible topic publishers | 1 (`edgetam_pointcloud_tracker`) |
| Refined safety corrections | 0 |

The local app published `/realtime_safety/camera/camera_info` at 320x240 with
the exact RGB stamp/frame and K=`[272,0,159.5; 0,272,119.5; 0,0,1]`. This is a
provisional profile projection, not checkerboard calibration.

Calling `/edgetam_tracker/set_enabled` with `false` returned success and
diagnostics reported `edge_enabled=false`, `state=disabled` while tracker FPS
continued. Calling it with `true` restored `state=ready` and nonzero Edge
latency without reloading the predictor. The UI uses the same service through
its non-blocking control bridge.

This proves live EdgeTAM inference, prompt/mask visualization, and fail-safe
runtime switching. Because `edge_refined_corrections` remained zero in this
scene, it does **not** prove accepted mask-to-cloud geometry improvement or
accuracy.

## Foreground/background correction follow-up

The first 8080 integration incorrectly treated the wooden panel as the sole
obstacle and generated a full-frame EdgeTAM prompt. Two independent blockers
were corrected:

- fixed-camera startup background calibration plus constrained frontal-plane
  removal now suppresses the wooden panel and stationary fixtures;
- sparse PointCloud2 depth support is graded against the projected 3D proposal
  rather than divided by the dense semantic-mask area.

Other guards reject prompts over 80% of the image, add four negative points,
retry the next official prompt API when a call succeeds but returns an empty
mask, and keep mask-result debug frames from being overwritten by prompt-only
frames.

Live observations after the gate correction but before the static baseline
was enabled showed one DEGRADED mask, zero INVALID masks, Edge latency
`62.084 ms`, and `34` accepted exact-context refined corrections. The target
was a stationary lower image-edge fixture, not a labelled hand. After an empty
workspace baseline (12 warmup + 16 calibration frames), the same static scene
reported background state `ready`, baseline `1885` points, about `2695` static
points removed per frame, and `processed=cluster=track=prompt=0` at `8.501 Hz`.
This validates background rejection and the correction path, but a physical
hand trial with ground truth is still required for hand recall/IoU claims.
