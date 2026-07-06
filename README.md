# social-safety-amr

AMR social-navigation safety analysis workbench for smooth real-image dataset video playback, official SAM 3 / SAM 3.1 segmentation and tracking integration, track motion estimation, BEV safety-map rendering, temporal VQA integration points, and deterministic safety decisions.

The current implementation is Phase 1.5 focused:

- Offline dataset playback with `michaelmunje/SocialNav-SUB` frame sequences from Hugging Face.
- Cached MP4 generation for image-sequence scenarios and HTML5 video playback in the GUI.
- CPU-only lightweight visual tracking fallback for fast BEV people/obstacle proposals when
  SAM 3.1 propagation is unavailable. This emits boxes and ground-contact points only; it
  never fabricates SAM segmentation masks.
- SCAND-style RGB-to-BEV inverse-perspective mapping (IPM) for imported MP4s using the
  Dataverse Azure Kinect intrinsics plus a configurable camera height/pitch estimate.
- Timestamp-synchronized `AnalysisPacket` metadata API for precomputed SAM 3 road masks, SAM 3.1 prompt-frame segmentation previews, propagated tracks when available, motion estimates, dynamic risk zones, robot corridor, and analysis delay status.
- Unified `FrameSource` and Pydantic v2 data models.
- FastAPI `dataset-service` with playback REST API, video endpoints, analysis endpoints, WebSocket playback metadata, SAM 3 client integration, precomputed video-analysis cache, and optional fallback road persistence.
- FastAPI `sam3-service` container image using the official `facebookresearch/sam3` package pinned to commit `5dd401d1c5c1d5c3eedff06d41b77af824517619`. It lazy-loads `facebook/sam3` for image/concept segmentation and `facebook/sam3.1` for multiplex video tracking.
- React/Vite TypeScript GUI with scenario selection, HTML5 video controls, SAM 3 road/agent overlays, prompt object or propagated track display, BEV Safety Map, dynamic risk zones when tracks exist, and explicit unavailable states.
- Docker Compose `dataset-demo` profile.
- Unit and integration tests for frame schemas, playback, cached MP4, road schemas, VQA JSON parsing, velocity estimation, direction fusion, dynamic risk zones, BEV transforms, stale analysis rejection, and API behavior.

RoboPoint, VQA, ROS 2, and full experiment execution remain scaffolded as independent services for later phases. Mock or fixture-only output is not used as formal experiment output. The local test fixture uses explicitly marked `fixture_color_segmentation` so the video/BEV data flow can be tested without claiming real SAM 3 or RoboPoint inference.

Current startup status on 2026-06-30: this host has driver `550.163.01`, CUDA compatibility `12.4`, and an RTX 4060 Ti with 16 GB VRAM. `cuda-full` Compose has started successfully with the host-compatible CUDA 12.4 / PyTorch 2.6 SAM3 image. The current authorized Hugging Face token can download and load both `facebook/sam3` and `facebook/sam3.1`. SAM 3 image segmentation is verified with true binary masks. SAM 3.1 can create a video session and return official object IDs plus true masks on the prompted frame, but this is only `SAM3 Prompt-Frame Segmentation Preview`. Cross-frame tracking is still unavailable because official multiplex propagation currently drops to an empty object batch after frame 0. Prompt-frame previews return `tracking_status="unavailable"` and are not reported as stable Track IDs. For interactive demos, the GUI now uses the lightweight CPU visual tracker when SAM 3.1 propagation has no cached result.

## Quick Start

```bash
make setup
make test
make demo-local
```

Open:

- GUI: http://localhost:5173
- Dataset service API: http://localhost:8000/docs

Docker dataset demo:

```bash
make demo-dataset
```

This profile does not require a camera, RealSense SDK, ROS 2, or `/dev/video`.

CUDA demo flow:

```bash
make doctor
make demo-cuda
```

Open http://localhost:5173 after `make demo-cuda`. With an authorized Hugging Face token, the service can load `facebook/sam3` and `facebook/sam3.1`; without one, the GUI still starts and plays SocialNav-SUB MP4s, but SAM 3 / SAM 3.1 model outputs remain unavailable.

After accepting the SAM 3 model terms and exporting `HF_TOKEN`, run:

```bash
make check-model-access
make download-models
make precompute-demo
```

SAM 3 / SAM 3.1 service:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full build sam3-service
HF_TOKEN=... docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full up sam3-service dataset-service web-gui
```

The CUDA profile uses:

- `SAM3_IMAGE_REPO=facebook/sam3`
- `SAM3_VIDEO_REPO=facebook/sam3.1`
- `HF_HOME=/models/huggingface`
- `SAM3_MAX_OBJECTS=6`
- `SAM3_USE_FA3=false`
- `SAM3_COMPILE=false`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

On the current RTX 4060 Ti host, the NVIDIA driver reports CUDA compatibility 12.4.
The SAM 3 CUDA image therefore uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
with PyTorch `2.6.0+cu124`. PyTorch `2.10.0+cu128` exists in the official cu128 index, but `nvidia/cuda:12.8.1` fails on this machine with `unsatisfied condition: cuda>=12.8`; upgrade the NVIDIA driver before switching this Dockerfile back to cu128.

`sam3-service` exposes `/segment-image` for road initialization plus persistent video tracking session endpoints:

- `POST /video/sessions`
- `POST /video/sessions/{session_id}/prompts`
- `POST /video/sessions/{session_id}/propagate`
- `GET /video/sessions/{session_id}/frames/{frame_index}`
- `GET /video/sessions/{session_id}/status`
- `DELETE /video/sessions/{session_id}`

Video sessions accept a complete absolute MP4 path and keep one official SAM 3.1 predictor session alive. Track IDs come from SAM 3.1 object IDs; the service does not synthesize `index + 1` IDs. Masks are returned as true binary-mask-derived COCO-style RLE and contour polygon data, with bounding boxes only as additional metadata.

Local continuous videos can be registered through:

- `POST /videos/import`
- `GET /videos/imports`
- `GET /videos/imports/{video_id}`
- `GET /videos/imports/{video_id}/analysis`
- `GET /videos/imports/{video_id}/frames/{frame_index}/image`
- `GET /videos/imports/{video_id}/bev-image`

`POST /videos/import` accepts either a JSON MP4 `path` visible to dataset-service or a multipart upload. It records native FPS, frame count, duration, width, height, SHA-256 hash, and the copied cache path under `/cache/dataset/imported_videos`. Imported MP4 analysis samples frames at 5 Hz, caches JPEG frames under `/cache/dataset/imported_videos/analysis_frames`, and runs the lightweight visual tracker for BEV boxes, approximate direction, approximate speed, and dynamic risk zones. Formal SAM 3.1 tracker demos should still use a real continuous MP4 of at least 10 seconds and 15 FPS, not SocialNav-SUB short image sequences or motion-interpolated playback MP4s.

For SCAND MP4s, `/bev-image` renders a 1000x420 top-down RGB ground-plane projection. The default profile is `scand_azure_kinect_estimated_ipm`, with Azure Kinect intrinsics from the SCAND Dataverse README and estimated extrinsics `camera_height_m=0.85`, `pitch_down_deg=24.0`, lateral range `[-4, 4] m`, and forward range `[0, 12] m`. Because the SCAND documentation states the Azure Kinect extrinsics were not recorded, this is an estimated IPM view until LiDAR/odometry calibration is added.

## GUI Workflow

The browser GUI is organized into three areas:

- Left: dataset, scenario, local MP4 import, load, play/pause, seek, playback speed, prediction horizon, VQA interval, overlay toggles, and explicit SAM 3 fallback editor controls.
- Center: continuous HTML5 video overlay and BEV Safety Map.
- Right: track list with direction, approximate speed, path relation, VQA status, risk level, and a collapsed debug JSON drawer.

When the dataset provides only image frames, `/scenarios/{scenario_id}/video-info` builds a cached playback MP4 with `ffmpeg`, and `/scenarios/{scenario_id}/video` serves that MP4. GUI playback uses a smooth 25 FPS motion-interpolated MP4 cache so sparse SocialNav-SUB frame sequences feel less like a slide show. Analysis metadata is fetched separately from `/scenarios/{scenario_id}/analysis` using `video.currentTime`; GUI requests read cached timestamp results and must not rerun full video inference.

The GUI also has a `Local MP4` control for importing and playing a real continuous MP4 through `/videos/import`. Selecting an imported MP4 runs lightweight CPU tracking against cached 5 Hz frames while the browser video continues playing. These overlays are labeled `Lightweight visual` and `Lightweight boxes`; they are useful for fast relative-position BEV checks, but they are not SAM masks and are not formal model output.

When the active video has an estimated camera profile, the BEV Safety Map draws the `/bev-image` RGB top-down raster behind the robot corridor, risk zones, tracks, and arrows. Track BEV positions use the same backend IPM projection via `track.metadata.bev_ground_point`, so the points and the raster share one camera model.

Real SAM 3.1 tracking is started explicitly through `POST /scenarios/{scenario_id}/analysis/sam3-video`. The dataset service builds a separate non-interpolated 640x360, 5 Hz, batch-size-1 analysis MP4 under the shared dataset cache, starts a SAM 3.1 video session, adds prompts such as `person`, and stores propagated frame results for later `/analysis` reads. If only the prompt frame succeeds, the result is labeled `SAM3 Prompt-Frame Segmentation Preview`, `tracking_status` remains `unavailable`, no motion or dynamic risk prediction is emitted, and the GUI does not copy those masks to later timestamps. SocialNav-SUB frames contain an RGB view stacked with a BEV map, so the SAM 3.1 analysis MP4 uses the top 16:9 RGB crop instead of resizing the whole composite frame. Switching scenarios closes the old SAM 3.1 session and clears cached tracking results.

SAM 3 road segmentation is started explicitly through `POST /scenarios/{scenario_id}/analysis/sam3-road`. The GUI `Re-run Road` button calls this endpoint with road/walkable-path prompts, stores a true-mask result if SAM 3 returns one, and reports the exact gated-access or missing-token blocker if the checkpoint cannot be loaded.

SAM 3 road segmentation is the primary path source. If SAM 3 road grounding is unavailable, the GUI shows `Road segmentation unavailable`, does not draw an auto-generated road polygon, and allows a saved fallback path only after the fallback editor is explicitly unlocked. The fallback is labeled `manual_fallback` and is not a formal SAM 3 result.

## SAM 3 Verification Status

Verified locally on 2026-06-29 and rechecked with authorized SAM checkpoints on 2026-06-30:

- `docker build -f docker/Dockerfile.sam3.cuda -t social-safety-amr-sam3-service .`: passed.
- Container CUDA check: PyTorch `2.6.0+cu124`, CUDA runtime `12.4`, `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 4060 Ti`, VRAM `16848453632` bytes, official SAM3 source commit `5dd401d1c5c1d5c3eedff06d41b77af824517619`, `Sam3RuntimeManager` import passed.
- `docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full up -d --build`: passed; dataset-service and sam3-service became healthy, web-gui started, GUI returned HTTP 200.
- `GET /scenarios/101_Spot_1_155/video-info`: passed after adding `ffmpeg` to the dataset image; generated/cached MP4 metadata returned 200.
- `GET /scenarios/101_Spot_1_155/video` byte-range read: passed and returned MP4 data.
- `make check-model-access`: passed for required SAM checkpoints with the current authorized token.
- `facebook/sam3` image checkpoint loaded from `/models/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`, revision `3c879f39826c281e95690f02c7821c4de09afae7`.
- `facebook/sam3.1` multiplex checkpoint loaded from `/models/huggingface/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt`, revision `daa63191845a41281374e725f4c9e51c7a824460`.
- `POST /scenarios/101_Spot_1_155/analysis/sam3-road`: passed; returned true binary-mask-derived SAM 3 detections with RLE, contour polygons, mask area, and bounding boxes only as metadata.
- `POST /scenarios/129_Spot_6_21/analysis/sam3-video`: created a SAM 3.1 video session and `person` prompt produced six official object IDs with true masks on frame 0.
- `POST /scenarios/34_Spot_0_765/analysis/sam3-video`: created a SAM 3.1 video session and returned six visible prompt-frame masks at timestamp `0.0`; this is labeled `SAM3 Prompt-Frame Segmentation Preview` and returns tracking unavailable until propagation succeeds.
- `GET /scenarios/34_Spot_0_765/analysis?timestamp_sec=0`: returned six prompt-frame objects with `tracking_status="unavailable"`, `tracking_fps=0.0`, `formal_model_output=false`, no motion estimates, and no dynamic risk zones.
- `GET /scenarios/34_Spot_0_765/analysis?timestamp_sec=0.6`: returned zero tracks and did not copy prompt-frame masks to the later timestamp.
- Latest SAM3.1 propagation smoke on 2026-06-30 remains degraded; observed errors include `Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same` and `index 26 is out of bounds for dimension 0 with size 26`.
- `POST /videos/import` multipart live smoke passed with a generated 15 FPS 320x180 MP4 and returned native FPS, frame count, duration, dimensions, hash, cache path, and video reference.
- Lightweight SocialNav-SUB live smoke after the latest rebuild returned tracks for `101_Spot_1_155`, `34_Spot_0_765`, `34_Spot_0_1070`, `129_Spot_7_157`, and `123_Spot_0_310`.
- Local MP4 lightweight live smoke passed with an uploaded 10 FPS 320x180 MP4; `/videos/imports/4d68a1f141a84e81/analysis?timestamp_sec=0.8` returned one `person` track, one motion estimate, `tracking_mode=lightweight_visual_tracker`, and no mask polygon.
- SCAND Dataverse live smoke passed using `A_Jackal_GDC_GDC_Fri_Oct_29_11.mp4` from `doi:10.18738/T8/0PRYRH`, Dataverse file id `133225`. It imported as `video_id=96dfe2562fed7f0f`, width `1280`, height `720`, native FPS `60`, and `/videos/imports/96dfe2562fed7f0f/bev-image?timestamp_sec=1.0` produced a real RGB IPM PNG of size `1000x420`.
- GUI sample scenarios now prioritize `34_Spot_0_765`, `101_Spot_1_155`, `34_Spot_0_1070`, `129_Spot_7_157`, and `123_Spot_0_310`, with `129_Spot_6_21` and `45_Spot_0_46` kept as extra candidates; their smooth playback MP4s were generated locally on 2026-06-30.
- SAM 3.1 propagation across 10 SocialNav-SUB frames is not accepted yet; official multiplex propagation currently returns an empty object batch after frame 0 and can fail with a `B=0` tensor expansion error. The service reports degraded status and does not fabricate fallback segmentation.
- `docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full config`: passed and shows `hf_cache:/models/huggingface` plus `dataset_cache:/cache/dataset`.
- `docker run --rm -v "$PWD":/app -w /app python:3.10-slim ... ruff check ... && pytest -q tests/unit/test_lightweight_tracker.py tests/unit/test_sam3_analysis_cache.py tests/integration/test_dataset_service.py`: passed, 12 tests on 2026-06-30.
- `/tmp/social-safety-venv/bin/ruff check .`: passed.
- `/tmp/social-safety-venv/bin/mypy packages services scripts`: passed.
- `/tmp/social-safety-venv/bin/python -m compileall services packages scripts tests`: passed.
- Frontend build passed in `node:20`.
- `npm --prefix apps/web run build` passed in `node:20` on 2026-06-30.

Reproduction and smoke targets:

- `make tracker-repro VIDEO=/cache/dataset/imported_videos/files/<video_id>.mp4`
- `make tracker-smoke VIDEO=/cache/dataset/imported_videos/files/<video_id>.mp4`
- `make demo-real VIDEO=/absolute/path/to/real_continuous.mp4`
- `make road-tracker-smoke` exits blocked until cross-frame propagation works.
- `make vqa-smoke` exits blocked because temporal VQA is not implemented.

Not verified in this environment:

- A SocialNav-SUB video with stable SAM 3.1 object IDs across at least 10 propagated frames. Do not treat the SAM 3.1 video integration as formally accepted until that run has completed.

Checkpoint configuration:

- Image/concept segmentation: repo `facebook/sam3`, checkpoint file `sam3.pt`, revision `main` unless overridden.
- Multi-object video tracking: repo `facebook/sam3.1`, checkpoint file `sam3.1_multiplex.pt`, revision `main` unless overridden.
- Hugging Face cache: `/models/huggingface`.

## Dataset

The default dataset source is Hugging Face `michaelmunje/SocialNav-SUB`. The adapter lazily downloads scenario frames into the Hugging Face cache and records the configured dataset revision in each `FramePacket`.

For CI and unit tests, the adapter can use a local mirror path via `SOCIALNAV_LOCAL_REPO`. Test fixtures are clearly separated from formal experiment outputs.

## Safety Policy

VQA/VLM components never directly command the AMR. The deterministic safety stack prioritizes:

1. Hard safety rules
2. Tracking and geometry
3. Time-to-zone prediction
4. VQA semantic reasoning
5. Navigation recommendation

Any uncertainty that affects safety must fail safe to `pause` or `human_review`; the system must not auto-return `continue` from insufficient information.

## Current Status

See [STATUS.md](STATUS.md).
