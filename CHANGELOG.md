# CHANGELOG

## 0.1.8 - 2026-06-30

- Added SCAND-style RGB-to-BEV inverse-perspective mapping with `scand_azure_kinect_estimated_ipm`.
- Added `/videos/imports/{video_id}/bev-image` and `/scenarios/{scenario_id}/bev-image` to render RGB ground-plane top-down PNGs.
- Changed backend BEV track conversion to attach `track.metadata.bev_ground_point`; GUI track dots now use backend IPM coordinates when available.
- Added BEV raster background rendering in the GUI when `robot_corridor.metadata.camera_profile` is active.
- Downloaded and imported SCAND Dataverse sample `A_Jackal_GDC_GDC_Fri_Oct_29_11.mp4` from DOI `10.18738/T8/0PRYRH`, file id `133225`, as `video_id=96dfe2562fed7f0f`.
- Verification: SCAND `/bev-image?timestamp_sec=1.0` returned a 1000x420 RGB PNG; SCAND `/analysis?timestamp_sec=1.0` returned 5 lightweight tracks, 2 motion estimates, and `camera_profile=scand_azure_kinect_estimated_ipm`.

## 0.1.7 - 2026-06-30

- Added CPU-only `LightweightVisionTracker` for fast non-formal people/obstacle proposals when SAM 3.1 propagation is unavailable.
- Integrated lightweight tracks into `AnalysisBuilder` so BEV positions, approximate motion, and dynamic risk zones can update without rerunning SAM on GUI requests.
- Added top-16:9 visual-height BEV mapping for SocialNav-SUB composite frames.
- Added imported MP4 lightweight analysis endpoints: `GET /videos/imports/{video_id}/analysis` and `GET /videos/imports/{video_id}/frames/{frame_index}/image`.
- Updated Local MP4 playback so imported continuous videos fetch lightweight 5 Hz analysis overlays while HTML5 video playback continues.
- Updated GUI status labels from SAM-only wording to `Tracker: Lightweight visual`, `Segmentation: Lightweight boxes`, and `Object tracking` when the fast path is active.
- Prioritized GUI sample scenarios that produce live lightweight tracks: `34_Spot_0_765`, `101_Spot_1_155`, `34_Spot_0_1070`, `129_Spot_7_157`, and `123_Spot_0_310`.
- Verification: Python ruff plus 12 focused unit/integration tests passed in `python:3.10-slim`; frontend build passed in `node:20`; live SocialNav-SUB and Local MP4 lightweight smokes returned tracks without fake masks.

## 0.1.6 - 2026-06-30

- Changed dataset-service status semantics so SAM3.1 prompt-frame-only output is labeled `SAM3 Prompt-Frame Segmentation Preview`, returns `tracking_status="unavailable"`, sets `formal_model_output=false`, and does not emit motion estimates or dynamic risk zones.
- Removed the GUI `sam3Preview` fallback that kept prompt-frame masks visible at later timestamps after propagation failed.
- Changed GUI labels so prompt-frame object IDs are shown as Object IDs, not stable Track IDs.
- Changed unavailable VQA display to `VQA unavailable`; geometry cues are no longer presented as VQA.
- Added `POST /videos/import`, `GET /videos/imports`, and `GET /videos/imports/{video_id}` for Local Continuous Video MP4 registration with native FPS, frame count, duration, dimensions, cache path, and SHA-256 hash.
- Added GUI Local MP4 import/playback for registered continuous videos without fabricating analysis overlays; lightweight analysis overlays were added later in `0.1.7`.
- Added `scripts/reproduce_sam31_tracking.py` for minimal official SAM3.1 reproduction on a real continuous MP4, preserving B=0 tracebacks under `outputs/sam31_reproduction/`.
- Added `tracker-repro`, `tracker-smoke`, `road-tracker-smoke`, `vqa-smoke`, and `demo-real` Make targets.
- Reconfirmed that the current host driver `550.163.01` cannot launch CUDA 12.8/cu128 containers; the host-compatible SAM3 Dockerfile remains CUDA 12.4 / PyTorch 2.6 until the NVIDIA driver is upgraded.
- Verification: Python ruff plus 10 focused unit/integration tests passed in `python:3.10-slim`; frontend build passed in `node:20`.

## 0.1.5 - 2026-06-30

- Verified the current authorized Hugging Face token can access and load both required SAM checkpoints.
- Recorded `facebook/sam3` revision `3c879f39826c281e95690f02c7821c4de09afae7` and checkpoint `/models/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`.
- Recorded `facebook/sam3.1` revision `daa63191845a41281374e725f4c9e51c7a824460` and checkpoint `/models/huggingface/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt`.
- Fixed SAM 3 image inference dtype handling with CUDA bfloat16 autocast and safe lower-precision tensor conversion.
- Verified `POST /scenarios/101_Spot_1_155/analysis/sam3-road` returns true binary-mask-derived SAM 3 detections with RLE, contour polygons, mask area, and bounding boxes only as metadata.
- Added top 16:9 RGB crop generation for SocialNav-SUB SAM 3.1 tracking MP4s so the tracker does not receive a downscaled RGB+BEV composite frame.
- Fixed SAM 3.1 predictor session startup against the official multiplex predictor signature.
- Aligned the SAM 3.1 vision backbone dtype for FA3-disabled RTX 4060-safe inference.
- Verified `POST /scenarios/129_Spot_6_21/analysis/sam3-video` creates a SAM 3.1 video session and returns six official object IDs with true masks on prompted frame 0.
- Fixed GUI visibility for partial SAM 3.1 results: degraded propagation responses now report the cached prompt-frame masks, the GUI seeks to that prompt frame, and dataset-service maps 640x360 top-crop mask coordinates back onto the original SocialNav-SUB display frame.
- Verified `34_Spot_0_765` returns six visible SAM 3.1 prompt-frame object masks at timestamp `0.0` after `Re-run Analysis`.
- Added SAM3 sample-scenario shortcuts in the GUI and made `34_Spot_0_765` the default scenario when available.
- Changed playback MP4 generation to create smooth 25 FPS ffmpeg motion-interpolated cache files for GUI playback; SAM 3.1 analysis MP4s remain non-interpolated 5 Hz top-crop clips.
- Pre-cached smooth playback MP4s for `34_Spot_0_765`, `129_Spot_6_21`, `101_Spot_1_155`, `34_Spot_0_1070`, `129_Spot_7_157`, and `45_Spot_0_46`.
- Kept SAM 3.1 propagation in degraded status because official multiplex propagation currently reaches an empty object batch after frame 0; no fake track IDs or fake rectangle segmentation are returned.
- Re-ran focused verification: `tests/unit/test_sam3_runtime.py`, `tests/unit/test_sam3_analysis_cache.py`, and ruff checks for the touched runtime and dataset-service files.

## 0.1.4 - 2026-06-30

- Added `ffmpeg` to the dataset-service Docker image so SocialNav-SUB image sequences can be converted into cached MP4 playback assets inside Compose.
- Changed SAM3 `/health` and `/readiness` to report `ok` while models are lazy-load idle; model load failures still return degraded request errors without fake segmentation.
- Added `POST /scenarios/{scenario_id}/analysis/sam3-road` so the GUI `Re-run Road` action calls SAM 3 image/concept segmentation instead of incorrectly reusing video tracking.
- Stored SAM 3 road segmentation results separately from SAM 3.1 video tracking results, then merged them for timestamp analysis without overwriting true masks or track IDs.
- Propagated SAM 3 gated-access and missing-token blockers into `AnalysisPacket.metadata.sam3_message` and `system_status.message` so the GUI reports the real reason segmentation is unavailable.
- Verified `cuda-full` Compose startup: dataset-service and sam3-service healthy, web GUI served at `http://localhost:5173`, and SAM3 CUDA runtime reports NVIDIA GeForce RTX 4060 Ti with PyTorch `2.6.0+cu124`.
- Verified `/scenarios/101_Spot_1_155/video-info` now returns 200 and `/video` serves an MP4 byte range.
- Verified `POST /scenarios/101_Spot_1_155/analysis/sam3-road` reaches SAM 3 image model loading and returns degraded with Hugging Face gated `facebook/sam3` 401 access instead of silently showing only BEV fallback.
- Re-ran `make precompute-demo`; it now reaches SAM3.1 load and records a blocked manifest due to Hugging Face gated `facebook/sam3.1` 401 access instead of failing at service startup or MP4 generation.

## 0.1.3 - 2026-06-29

- Added `make doctor` and `scripts/gpu_doctor.sh`, generating host/GPU/Docker/CUDA JSON diagnostics under `outputs/doctor/`.
- Diagnosed the CUDA mismatch: host driver `550.163.01` supports CUDA `12.4`; CUDA `12.8` and `12.6` containers fail, while CUDA `12.4.1` containers pass.
- Changed the SAM 3 CUDA image to `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, Python 3.12, PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`, and torchaudio `2.6.0+cu124`.
- Verified Docker PyTorch CUDA on the real GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB VRAM, `torch.cuda.is_available() == True`.
- Added real Hugging Face access/download scripts and manifest output: `scripts/check_model_access.py`, `scripts/download_models.py`, `outputs/model_access.json`, and `outputs/model_manifest.json`.
- Downloaded SmolVLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; SAM 3 and SAM 3.1 remain blocked by Hugging Face gated access without `HF_TOKEN`.
- Added `scripts/precompute_scenario.py`, `make precompute-scenario`, and `make precompute-demo`; blocked precompute writes manifest/logs instead of fabricating output.
- Added SAM 3 video session status API and expanded `/video/sessions` to accept `video_path`, `scenario_id`, `analysis_fps`, and `max_objects`.
- Added `/readiness` and `/runtime-info` endpoints to SAM3, dataset, RoboPoint, and VQA services.
- Updated the GUI with precompute status, re-run analysis/road buttons, dataset revision, precomputed/live status, and formal-output status.
- Verified: unit tests, ruff, mypy, compileall, SAM3 Docker build, GPU doctor, frontend lint, and frontend build. Formal SAM 3 / SAM 3.1 inference and E2E remain blocked by gated model access.

## 0.1.2 - 2026-06-29

- Replaced the incorrect Transformers `mask-generation` SAM 3 integration with the official `facebookresearch/sam3` package pinned to commit `5dd401d1c5c1d5c3eedff06d41b77af824517619`.
- Rebuilt the SAM 3 CUDA image on Python 3.12 with PyTorch `2.10.0+cu128`, Hugging Face cache at `/models/huggingface`, and RTX 4060-oriented defaults that disable flash-attn-3, `torch.compile`, and warm-up compilation.
- Added lazy `Sam3ImageRuntime`, `Sam31VideoRuntime`, and `Sam3RuntimeManager` classes with explicit load/unload, CUDA cache clearing, VRAM reporting, checkpoint source reporting, and degraded OOM handling.
- Added persistent SAM 3.1 video session endpoints and dataset-service client methods for session start, prompts, propagation, frame-result reads, and close.
- Removed fake `index + 1` track IDs; video detections now require official SAM 3.1 object IDs.
- Preserved true binary masks as RLE and mask-derived polygon data instead of replacing segmentation with bounding-box rectangles.
- Changed dataset analysis so GUI timestamp requests read precomputed SAM 3.1 frame results and do not rerun full tracking inference.
- Added 640x360, 5 Hz, max-6-object video-analysis preparation and scenario-switch session cleanup.
- Added SAM 3 runtime/cache unit tests and gated real-CUDA integration tests for container CUDA, image checkpoint load, video checkpoint load, real masks, stable object IDs, OOM resilience, and playback non-blocking behavior.
- Verification on this machine: Docker image build passed; container import showed PyTorch `2.10.0+cu128`, CUDA runtime `12.8`, SAM3 commit `5dd401d1c5c1d5c3eedff06d41b77af824517619`; unit tests passed 28 tests; ruff and compileall passed.
- Not yet verified on this machine: actual GPU/VRAM, `facebook/sam3` checkpoint load, `facebook/sam3.1` checkpoint load, and SocialNav-SUB cross-frame person tracking, because `nvidia-smi` could not communicate with the NVIDIA driver and `docker run --gpus all` failed with `unsatisfied condition: cuda>=12.8`.

## 0.1.1 - 2026-06-29

- Reworked the GUI around HTML5 video playback instead of frame-by-frame image updates.
- Added cached MP4 generation for image-sequence scenarios and video metadata endpoints.
- Added unified `AnalysisPacket` schemas for road segmentation, tracks, motion estimates, VQA direction status, robot corridor, dynamic risk zones, and system delay status.
- Added a containerized `sam3-service` adapter and dataset-service SAM 3 client. SAM 3 failures are reported as degraded/unavailable instead of replaced by fabricated masks.
- Replaced the primary manual danger-zone workflow with SAM 3 road segmentation, explicit fallback editor controls, and a `manual_fallback` road polygon API.
- Added BEV Safety Map rendering with robot corridor, optional SAM 3 road/corridor overlays, and red dynamic risk zones.
- Added geometry helpers and tests for ground-contact point, timestamp velocity, stationary detection, direction labels, VQA JSON parsing, direction fusion, constant-velocity prediction, swept corridors, intersections, BEV transform, and stale analysis rejection.
- Marked fixture-only color segmentation as non-formal output instead of presenting it as SAM 3/RoboPoint inference.
- Verified the SAM 3 image builds, while the local model runtime remains blocked by gated Hugging Face access for `facebook/sam3` and missing Docker NVIDIA runtime support.

## 0.1.0 - 2026-06-29

- Created Phase 0/1 project scaffold.
- Added dataset playback service, shared models, frame source adapters, GUI, Docker profile, tests, and documentation.
