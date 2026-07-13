# Real-Time 3D Obstacle Segmentation and Safety Decision System

Live RGB-video instance segmentation, stable 2D/3D tracking, monocular or St4RTrack reconstruction, directional swept danger volumes, local forward-corridor planning, and a mouse-interactive Viser dashboard.

The pipeline is designed around one safety rule: stale frames are less useful than current frames. Every inter-worker queue is bounded (default size 2), drops its oldest item under pressure, and records the drop. Models run outside the GUI renderer, safety prediction continues at its own target rate, and no complete-video preprocessing or per-frame disk round-trip is used.

## What is implemented

- MP4, AVI, MOV, MKV, WEBM, webcam, and RTSP sources, including pause, resume, restart, loop, seek, and playback speed.
- GUI upload and command-line source selection.
- Persistent YOLO11n-seg CUDA backend with mixed precision; COCO obstacle classes are normalized to person, bicycle, chair, bag, suitcase, vehicle, and related obstacle types.
- Timestamp-aware stable 2D IDs, short-occlusion association, and between-segmentation prediction.
- Real Depth Anything V2 Small relative monocular depth point maps; no synthetic point cloud fallback.
- Optional in-memory St4RTrack frame-pair adapter. It directly returns the tracking/reconstruction point maps and never writes NPY files.
- Mask-to-pointmap 3D clusters plus radius-connected `unknown_obstacle` clustering.
- Constant-velocity 3D Kalman filters with timestamp-derived velocity and static/dynamic hysteresis.
- Three-second trajectory prediction and an uncertainty-inflated series-of-ellipsoids swept volume.
- RANSAC ground plane, traversable-region filtering, nine local path candidates, collision rejection, detour selection, and all-paths-blocked STOP.
- SAFE, CAUTION, WARNING, STOP, and DEGRADED with immediate STOP and debounced release.
- Persistent Viser scene handles for point clouds, bounding boxes, arrows, trails, future paths, red/orange danger volumes, green traversable ground, candidates, and the selected cyan path. Updating a frame never writes client camera state, so the viewing angle is preserved.
- Actual input/display/segmentation/3D/safety rates, avg/p95 latency, drops, queue usage, RAM, and allocated VRAM.
- Streaming JSONL, trajectory CSV, optional annotated MP4 and final binary PLY.
- A benchmark report with measured rates and RAM/VRAM trend estimates.
- Optional ROS 2 JSONL bridge.

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
git clone <this-repository>
cd realtime_3d_safety_decision
bash scripts/setup.sh
source .venv/bin/activate
bash scripts/download_models.sh
```

Or install into an existing environment:

```bash
python3 -m pip install -r requirements.txt
bash scripts/download_models.sh
```

TensorRT is optional and never required to start:

```bash
python3 -m pip install -r requirements_tensorrt.txt
```

The AGX profile expects an exported `yolo11n-seg.engine`. Export it on the target architecture with Ultralytics before selecting that profile.

## Optional St4RTrack

```bash
bash scripts/download_models.sh --st4rtrack
```

This shallow-clones the official repository into ignored `third_party/St4RTrack`. On first use, the adapter loads `yupengchengg147/St4RTrack` from Hugging Face unless `st4rtrack_checkpoint` points to a local file. The external code/model has a non-commercial scientific research license; review it before use.

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

## Run

Fast profile:

```bash
python app.py --source /path/to/test.webm --profile realtime_fast --device cuda
# or
bash scripts/run_fast.sh /path/to/test.webm
```

Webcam:

```bash
python app.py --source 0 --profile realtime_fast --device cuda
```

Hybrid:

```bash
python app.py --source /path/to/test.mp4 --profile realtime_fast --depth-mode hybrid --device cuda
```

AGX:

```bash
python app.py --source /path/to/test.mp4 --profile agx --device cuda
```

GUI upload only:

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

| Profile | Segmentation | 3D input | Maximum displayed points | Intended use |
|---|---:|---:|---:|---|
| `realtime_fast` | 320 | 224 | 30,000 | Default latency-first mode |
| `realtime_balanced` | 416 | 320 | 60,000 | More 3D detail |
| `realtime_quality` | 640 | 512 | 100,000 | Visualization quality; may miss 10 Hz |
| `agx` | 320 TensorRT | 224 | 20,000 | Serialized low-memory Jetson mode |

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

- A single monocular RGB stream has no inherent metric scale. Relative depth also has temporal scale/shift noise; the Kalman filter and St4RTrack corrections reduce but do not eliminate it.
- The provided St4RTrack adapter is pairwise feed-forward. Upstream test-time adaptation is intentionally excluded from the real-time path.
- YOLO11n-seg uses COCO classes and cannot natively distinguish every requested fine-grained class (for example wheelchair versus chair); unmatched geometric clusters are `unknown_obstacle`.
- Unknown 3D clustering is intentionally low-rate and conservative to protect safety latency.
- Camera motion is estimated as robust 2D affine motion, not a full metric 6-DoF pose.
- The local planner is a `DEMO FORWARD CORRIDOR`, not Nav2 or a replacement for a certified robot safety controller.
- TensorRT, RGB-D camera capture, actual AGX hardware, and the official St4RTrack checkpoint require target-specific assets/hardware and are not silently reported as tested.
- Viser supports the requested interaction and persistent view, but uploaded videos are copied to a bounded per-process temporary file and removed by the operating system's temp cleanup policy.

## References

- [Official St4RTrack repository](https://github.com/HavenFeng/St4RTrack)
- [St4RTrack project page](https://st4rtrack.github.io/)
- [St4RTrack paper](https://arxiv.org/abs/2504.13152)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Viser](https://viser.studio/)
