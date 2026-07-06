# STATUS

Last updated: 2026-06-30

## Phase Status

- Phase 1 GPU doctor and Docker CUDA: completed.
- Phase 2 model access and download: completed for SAM 3 / SAM 3.1 with the current authorized Hugging Face token.
- Phase 3 real SAM 3 image inference: completed for road initialization; true binary-mask-derived detections are returned.
- Phase 4 real SAM 3.1 video tracking: blocked / incomplete; the checkpoint loads and prompted frame 0 returns official object IDs and true masks, but cross-frame propagation is not accepted yet.
- Phase 5 offline precompute pipeline: partially completed; API paths and cache separation exist, but SAM 3.1 propagation currently returns degraded instead of formal tracks.
- Phase 5.5 lightweight visual BEV demo path: completed as non-formal output; SocialNav-SUB and imported MP4s can produce fast CPU boxes, approximate BEV positions, approximate motion, and risk zones without claiming SAM masks.
- Phase 5.6 SCAND RGB-to-BEV IPM: completed as estimated non-formal output; imported SCAND MP4s can render RGB top-down ground-plane PNGs and align track points to the same projection.
- Phase 6 RoboPoint: not started; access metadata is visible, but 13B/LoRA execution is not integrated.
- Phase 7 Temporal VQA: not implemented; geometry output is not treated as VQA.
- Phase 8 motion/risk zones: available for fixture/lightweight non-formal tracks; formal motion/risk remains blocked by missing stable SAM 3.1 tracks.
- Phase 9 GUI: operational in `cuda-full` Compose and reads timestamp analysis caches without rerunning full inference on GUI requests.
- Formal E2E: not complete until SAM 3.1 produces stable official object IDs across at least 30 frames of a real continuous MP4 and the road/VQA/risk stack consumes those tracks.

## Required Honesty Labels

- Prompt-frame segmentation: completed.
- Cross-frame SAM3.1 tracking: blocked / incomplete.
- Temporal VQA: not implemented.
- Dynamic risk prediction: available only for non-formal lightweight/fixture tracks; formal SAM 3.1 risk prediction is blocked by missing stable propagated tracks.
- GUI status for prompt-frame masks: `SAM3 Prompt-Frame Segmentation Preview`.
- GUI/API tracking status for prompt-frame-only output: `tracking_status="unavailable"`.
- GUI status for the fast fallback: `tracking_mode="lightweight_visual_tracker"`, `segmentation_status="not_used_lightweight_boxes"`, `formal_model_output=false`.
- GUI BEV raster for SCAND/imported MP4: `robot_corridor.metadata.camera_profile="scand_azure_kinect_estimated_ipm"`.

## Current Verified Hardware

- GPU: NVIDIA GeForce RTX 4060 Ti
- VRAM: 16380 MiB, `16848453632` bytes
- Driver: `550.163.01`
- Host CUDA compatibility: `12.4`
- Container PyTorch: `2.6.0+cu124`
- Container CUDA runtime: `12.4`
- Secure Boot: disabled
- NVIDIA Container Toolkit: installed
- Docker NVIDIA runtime/CDI: available

The requested CUDA 12.8 / PyTorch cu128 path is not runnable on this host driver. Verified on 2026-06-30: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` fails before container startup with `unsatisfied condition: cuda>=12.8`. The official PyTorch cu128 index contains `torch==2.10.0+cu128`, but this host needs a newer NVIDIA driver before that image can launch. The SAM 3 CUDA image therefore remains on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` and PyTorch `2.6.0+cu124` for actual startup on this machine.

## SAM Checkpoints

- SAM 3 package source: `facebookresearch/sam3`
- SAM 3 package commit: `5dd401d1c5c1d5c3eedff06d41b77af824517619`
- Image model repo: `facebook/sam3`
- Image model revision: `3c879f39826c281e95690f02c7821c4de09afae7`
- Image checkpoint: `/models/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`
- Video model repo: `facebook/sam3.1`
- Video model revision: `daa63191845a41281374e725f4c9e51c7a824460`
- Video checkpoint: `/models/huggingface/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt`
- Hugging Face cache: `/models/huggingface`

## Completed In This Pass

- Added a CPU-only `LightweightVisionTracker` that extracts foreground boxes, maintains nearest-neighbor track IDs across frames, and emits ground-contact points for BEV mapping without creating fake masks.
- Integrated lightweight tracking into `AnalysisBuilder` when SAM 3.1 propagated results are unavailable.
- Added effective top-16:9 visual-height BEV mapping for SocialNav-SUB composite frames so image ground points map to the RGB view rather than the lower embedded BEV panel.
- Added Local MP4 lightweight analysis through `GET /videos/imports/{video_id}/analysis` and frame extraction through `GET /videos/imports/{video_id}/frames/{frame_index}/image`.
- Added RGB-to-BEV IPM rendering through `GET /videos/imports/{video_id}/bev-image` and `GET /scenarios/{scenario_id}/bev-image`.
- Added SCAND Azure Kinect estimated camera profile using Dataverse intrinsics and estimated extrinsics `camera_height_m=0.85`, `pitch_down_deg=24.0`.
- Added `track.metadata.bev_ground_point` so GUI track dots and backend risk zones use the same IPM projection.
- Added GUI BEV raster background when an estimated camera profile is active.
- Updated the GUI to show `Tracker: Lightweight visual`, `Segmentation: Lightweight boxes`, and `Object tracking` instead of treating every non-SAM frame as unavailable.
- Updated Local MP4 playback so imported videos fetch lightweight analysis while the browser video continues playing.
- Prioritized GUI sample scenarios with live lightweight tracks: `34_Spot_0_765`, `101_Spot_1_155`, `34_Spot_0_1070`, `129_Spot_7_157`, and `123_Spot_0_310`.
- Verified that the current authorized Hugging Face token can access and load both required SAM checkpoints.
- Replaced the stale gated-access status with real checkpoint revision and cache-path reporting.
- Fixed SAM 3 image inference dtype handling with CUDA bfloat16 autocast.
- Fixed tensor-to-NumPy conversion for lower-precision masks.
- Verified `POST /scenarios/101_Spot_1_155/analysis/sam3-road` returns real SAM 3 binary masks, RLE, contour polygons, mask area, and bounding boxes only as metadata.
- Added a top 16:9 RGB crop for SocialNav-SUB SAM 3.1 analysis MP4 generation so the tracker sees the camera image instead of a downscaled RGB+BEV composite.
- Fixed SAM 3.1 predictor session startup against the official multiplex predictor signature.
- Aligned the SAM 3.1 vision backbone dtype for FA3-disabled RTX 4060-safe inference.
- Added prompt-frame partial-result reporting labeled as `SAM3 Prompt-Frame Segmentation Preview`; prompt-frame output is segmentation only, not cross-frame tracking.
- Added coordinate mapping from 640x360 SAM 3.1 top-crop masks back to the original SocialNav-SUB display frame.
- Added GUI SAM3 sample-scenario shortcuts and defaulted the GUI to `34_Spot_0_765` when available.
- Removed the GUI fallback that kept showing prompt-frame masks at later timestamps.
- Added smooth 25 FPS motion-interpolated playback MP4 generation for display only; formal SAM 3.1 analysis still uses non-interpolated 5 Hz clips.
- Added `POST /videos/import` plus an imported-video registry for real Local Continuous Video MP4s with native FPS, frame count, duration, size, and SHA-256 hash.
- Added GUI Local MP4 import/playback for imported continuous videos; this now supports non-formal lightweight analysis overlays.
- Added `scripts/reproduce_sam31_tracking.py` to reproduce official SAM3.1 start-session, prompt, and propagation on a real continuous MP4, saving `input_info.json`, `session_log.jsonl`, `prompt_result.json`, `propagated_results.jsonl`, `overlay.mp4`, and `error_trace.txt`.
- Added Make targets `tracker-repro`, `tracker-smoke`, `road-tracker-smoke`, `vqa-smoke`, and `demo-real`.
- Pre-cached smooth playback MP4s for `34_Spot_0_765`, `129_Spot_6_21`, `101_Spot_1_155`, `34_Spot_0_1070`, `129_Spot_7_157`, and `45_Spot_0_46`.
- Ensured dataset-service closes any previous active SAM 3.1 session before starting a new scenario or retrying analysis.
- Confirmed SAM 3.1 `person` prompting on `129_Spot_6_21` returns six official object IDs with true masks on frame 0.
- Confirmed SAM 3.1 `person` prompting on `34_Spot_0_765` returns six visible prompt-frame object masks at timestamp `0.0`; these are not stable Track IDs.
- Preserved degraded behavior on SAM 3.1 propagation failure; no fake segmentation or rectangle fallback is returned as a model result.

## Current SAM 3.1 Propagation Blocker

The official SAM 3.1 multiplex checkpoint loads, and prompt frame 0 produces object IDs and masks. Propagation across SocialNav-SUB frames is still failing formal acceptance:

- Scenario: `129_Spot_6_21`
- Analysis MP4: `/cache/dataset/video_cache/michaelmunje_SocialNav-SUB/f750caf46e5b33e6aef8c95af6a92fb4aff1d1b1/129_Spot_6_21_640x360_top1.78_5fps.mp4`
- Prompt: `person`
- Frame 0 result: 6 official SAM 3.1 object IDs, true binary masks.
- Propagation failure: official multiplex propagation later reaches an empty object batch (`B=0`) and fails with a tensor expansion error.
- Latest 2026-06-30 live failure on the rebuilt service: `Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same`.
- Latest final smoke after web/file-upload rebuild: `index 26 is out of bounds for dimension 0 with size 26`.
- Point-prompt seeding experiments avoided the crash but returned empty object IDs after frame 0.

This means the repository now has real SAM 3 image segmentation and real SAM 3.1 prompt-frame segmentation, but it must not be reported as having completed cross-frame SAM 3.1 person tracking yet. The next required step is running `scripts/reproduce_sam31_tracking.py` on a true continuous MP4, not on SocialNav-SUB 10-frame image sequences.

## RTX 4060 Safety Settings

- Analysis resolution: 640x360
- Tracking target: 5 Hz
- Batch size: 1
- Max objects: 6
- Flash-attn-3: disabled
- `torch.compile`: disabled
- Warm-up compilation: disabled
- SAM 3 image model and SAM 3.1 video model are not kept resident at the same time.
- CUDA OOM handling unloads the active model, clears CUDA cache, and returns degraded status.
- GUI `/analysis` timestamp reads use cached results and must not rerun full tracking inference.

## Verification Completed

- Live lightweight SocialNav-SUB smoke after the latest rebuild: `101_Spot_1_155` returned 1 track, `34_Spot_0_765` returned 2 tracks, `34_Spot_0_1070` returned 1 track plus 1 motion estimate, `129_Spot_7_157` returned 2 tracks, and `123_Spot_0_310` returned 1 track. All reported `tracking_mode=lightweight_visual_tracker`.
- Live Local MP4 lightweight smoke after the latest rebuild: uploaded `moving-lightweight` 10 FPS 320x180 MP4, imported as `video_id=4d68a1f141a84e81`, and `GET /videos/imports/4d68a1f141a84e81/analysis?timestamp_sec=0.8` returned 1 `person` track, 1 motion estimate, `segmentation_status=not_used_lightweight_boxes`, and no mask polygon.
- Live SCAND Dataverse IPM smoke after the latest rebuild: downloaded `A_Jackal_GDC_GDC_Fri_Oct_29_11.mp4` from `doi:10.18738/T8/0PRYRH`, Dataverse file id `133225`, imported it as `video_id=96dfe2562fed7f0f`, and verified `GET /videos/imports/96dfe2562fed7f0f/bev-image?timestamp_sec=1.0` returned a 1000x420 RGB PNG. `GET /videos/imports/96dfe2562fed7f0f/analysis?timestamp_sec=1.0` returned 5 lightweight tracks, 2 motion estimates, and `camera_profile=scand_azure_kinect_estimated_ipm`.
- `make doctor`: passed.
- `docker build -f docker/Dockerfile.sam3.cuda -t social-safety-amr-sam3-service .`: passed.
- SAM3 container CUDA smoke: passed, `torch.cuda.is_available() == True`.
- `make check-model-access`: passed for `facebook/sam3` and `facebook/sam3.1` with the current authorized token.
- Direct SAM 3 image model load in container: passed, revision `3c879f39826c281e95690f02c7821c4de09afae7`.
- Direct SAM 3.1 video model load in container: passed, revision `daa63191845a41281374e725f4c9e51c7a824460`.
- `POST /scenarios/101_Spot_1_155/analysis/sam3-road`: passed with true SAM 3 masks.
- `POST /scenarios/129_Spot_6_21/analysis/sam3-video`: partially passed; session and frame 0 prompt masks succeeded, propagation returned degraded.
- `POST /scenarios/34_Spot_0_765/analysis/sam3-video`: partially passed after GUI visibility fix; response reported `prompt_result_available=true`, `prompt_detection_count=6`, and `cached_analysis_frames=1`.
- `GET /scenarios/34_Spot_0_765/analysis?timestamp_sec=0`: now returns six SAM 3.1 prompt-frame objects with mapped display-frame coordinates, `tracking_status=unavailable`, no motion estimates, no dynamic risk zones, and `formal_model_output=false`.
- `GET /scenarios/34_Spot_0_765/analysis?timestamp_sec=0.6`: returned zero tracks, `tracking_status=unavailable`, and no copied prompt-frame masks.
- `GET /scenarios/34_Spot_0_765/video-info`: returned `source=generated_smooth_mp4` after smooth playback cache regeneration; later sample scenario requests also generated smooth cached MP4s.
- `GET /video/sessions/{session_id}/frames/0`: returned six detections with official object IDs and non-rectangle binary masks during the SAM 3.1 prompt-frame test.
- `POST /videos/import` multipart live smoke: passed with a generated 15 FPS 320x180 MP4; returned `video_id=fd26b4d7f8ef78bc`, native FPS 15, 15 frames, duration 1.0 s, SHA-256 hash, and cached path `/cache/dataset/imported_videos/files/fd26b4d7f8ef78bc.mp4`.
- `GET /videos/imports/fd26b4d7f8ef78bc/video` byte-range read: passed, 128 bytes returned.
- `docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full up -d --build`: passed on 2026-06-30.
- Running service status after startup:
  - dataset-service: healthy, port 8000.
  - sam3-service: healthy, port 8020, CUDA visible, lazy-load idle when no model is resident; after the live prompt test the SAM3.1 video model was resident with revision `daa63191845a41281374e725f4c9e51c7a824460`.
  - web-gui: started, port 5173, HTTP 200.
  - robopoint-service: HTTP available, degraded stub because the runtime is not loaded.
  - vqa-service: HTTP available, degraded stub because the runtime is not loaded.
- `docker run --rm -v "$PWD":/app -w /app python:3.10-slim ... ruff check ... && pytest -q tests/unit/test_bev_projection.py tests/unit/test_lightweight_tracker.py tests/integration/test_dataset_service.py`: passed, 8 targeted tests on 2026-06-30.
- Earlier focused Python checks with SAM3 cache tests passed 12 tests on 2026-06-30.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/social-safety-venv/bin/python -m pytest -q tests/unit`: passed earlier, 28 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/social-safety-venv/bin/python -m pytest -q tests/integration`: passed earlier, 2 tests passed and 5 formal GPU/model tests skipped.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/social-safety-venv/bin/python -m pytest -q tests/e2e`: passed earlier with 1 formal E2E test skipped.
- `/tmp/social-safety-venv/bin/ruff check services/sam3_service/runtime.py services/dataset_service/video_cache.py services/dataset_service/analysis.py services/dataset_service/app.py`: passed.
- `/tmp/social-safety-venv/bin/ruff check .`: passed earlier.
- `/tmp/social-safety-venv/bin/mypy packages services scripts`: passed earlier.
- `/tmp/social-safety-venv/bin/python -m compileall services packages scripts tests`: passed earlier.
- Frontend build with `node:20`: passed on 2026-06-30.
- `docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full ps`: dataset-service and sam3-service healthy; web-gui up at port 5173.
- `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version`: `NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 5220 MiB, 550.163.01` after SAM3.1 prompt-frame smoke.
- `DELETE /models/video`: passed after the smoke test; SAM3.1 video model unloaded and `/runtime-info` reported `video_loaded=false`, VRAM used `1414266880` bytes, allocated `691788288` bytes, reserved `834666496` bytes.
- `GET /scenarios/101_Spot_1_155/video-info`: passed and returned cached MP4 metadata.
- `GET /scenarios/101_Spot_1_155/video` byte-range read: passed and returned MP4 data.

## Not Yet Accepted

- Stable SAM 3.1 object IDs across 10 SocialNav-SUB frames.
- Stable SAM 3.1 object IDs across at least 30 frames of a real continuous MP4.
- LiDAR/odometry-calibrated SCAND metric BEV; current RGB-to-BEV uses estimated camera extrinsics because the Dataverse README does not record Azure Kinect extrinsics.
- Full formal person/obstacle tracking output.
- Formal E2E experiment results using SAM 3.1 tracks.
- RoboPoint runtime integration.
- Temporal VQA runtime integration.

## Host Node Note

- Host `node` is `v12.22.9`, too old for the current Vite/ESLint toolchain.
- Frontend verification was performed with `node:20-bookworm`.

## Formal Experiment Guardrails

- Build success is not inference success.
- Import success is not model success.
- Checkpoint metadata visibility is not checkpoint download success.
- Prompt-frame segmentation is not cross-frame tracking success.
- No fallback may be presented as SAM 3, SAM 3.1, RoboPoint, or VQA formal output.
- Formal SAM 3.1 acceptance still requires real checkpoint load, real SocialNav-SUB propagation, stable official object IDs across at least 10 valid frames, and non-rectangular true mask evidence.
