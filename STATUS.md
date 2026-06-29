# STATUS

Last updated: 2026-06-29

## Phase 0: Repository Audit

Status: completed for an empty workspace.

- Existing repository: none detected; `/home/laiting/Desktop/NTHU_HRC_lab/itri_safty_descision` was not a git repository and contained no project files.
- Reusable RoboPoint code: none found.
- Reusable SAM/SAM 3 code: none found.
- Reusable ROS 2 package: none found.
- Reusable GUI/dashboard/Docker: none found.
- License risks: third-party datasets/models require explicit revision and license capture before formal experiments.
- Dependency risks: SAM 3, RoboPoint, Qwen, SmolVLM, ROS 2 Humble, CUDA, and Jetson builds may need incompatible Python/CUDA stacks, so they are isolated by service and Docker image.

## Phase 1: Dataset Playback MVP

Status: superseded by Phase 1.5 video/analysis GUI.

Implemented:

- Pydantic v2 shared models.
- `FrameSource` interface.
- SocialNav-SUB Hugging Face/local-mirror adapter.
- Deterministic playback manager with play, pause, stop, reset, seek, and step.
- FastAPI dataset-service with health, dataset/scenario/frame, playback, WebSocket, and zone APIs.
- React/Vite GUI for scenario selection, playback controls, image playback, overlay toggles, and manual danger-zone polygon editing.
- Zone save/load under `config/zones/<dataset>/<scenario>.json`.
- Docker dataset-demo profile.
- Unit and integration tests.

Verification completed on 2026-06-29:

- `make test`: passed, 13 unit tests and 1 integration test.
- `make lint`: passed.
- `make typecheck`: passed.
- `npm --prefix apps/web run lint`: passed.
- `npm --prefix apps/web run build`: passed.
- `SOCIALNAV_LOCAL_REPO=tests/fixtures/socialnav_sub make experiment-smoke`: passed for Phase 1 playback-only smoke.
- Hugging Face `michaelmunje/SocialNav-SUB` remote scenario listing: 60 scenarios.
- Remote frame lazy download: `101_Spot_1_155` frame 0 returned image metadata and PNG successfully.
- `docker compose --profile dataset-demo config`: passed.
- `docker compose --profile dataset-demo build`: passed for dataset-service and web-gui images.
- Docker rebuild after adding `.dockerignore`: passed with reduced build context.
- `docker compose --profile dataset-demo up -d`: passed; dataset-service became healthy, web-gui returned HTTP 200, and remote scenario listing returned 60 scenarios. Compose was shut down after validation.
- Local dev servers verified:
  - dataset-service: http://localhost:8000/health
  - web-gui: http://localhost:5173/

## Phase 1.5: Smooth Video and Dynamic Risk View

Status: implemented and locally verified on fixture data, with SAM 3 service adapter added and blocked by external model/runtime access on this machine.

Implemented:

- Cached MP4 generation from SocialNav-SUB image-sequence scenarios via `ffmpeg`.
- HTML5 video playback path through `/scenarios/{scenario_id}/video-info` and `/scenarios/{scenario_id}/video`.
- Timestamp-synchronized `/scenarios/{scenario_id}/analysis` API returning `AnalysisPacket`.
- Shared schemas for `RoadSegmentationResult`, `TrackObservation`, `MotionEstimate`, `VQADirectionEstimate`, `DynamicRiskZone`, `RobotCorridor`, and `AnalysisSystemStatus`.
- SAM 3 service API with `/health`, `/model-info`, and `/segment-image`.
- Dataset-service SAM 3 client. `/analysis` calls SAM 3 for real SocialNav-SUB frame images when `SAM3_SERVICE_URL` is configured.
- Road fallback alias endpoints under `/road/{scenario_id}`.
- Manual fallback path is saved as `manual_fallback`; the GUI no longer draws an auto-generated road polygon when SAM 3 is unavailable.
- BEV Safety Map with robot corridor, robot origin, current agent positions, direction arrows, predicted trajectories, red dynamic risk zones, and collision-risk labels when model outputs are available.
- Track table with Track ID, type, fused direction, approximate normalized speed, path relation, VQA status, and risk.
- Geometry utilities for ground-contact points, timestamp velocity, stationary detection, direction labels, VQA/geometry direction fusion, constant-velocity prediction, swept corridors, polygon intersection, approximate BEV transform, and stale analysis rejection.
- VQA JSON parser with fixed schema behavior; parse errors do not stop analysis.
- Fixture-only color segmentation for the local artificial test frames, explicitly marked `fixture_color_segmentation` and `formal_model_output=false`.

Verification completed on 2026-06-29:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`: passed, 23 tests.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/mypy packages services`: passed.
- `npm --prefix apps/web run lint`: passed.
- `npm --prefix apps/web run build`: passed.
- `docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full build sam3-service`: passed.
- `docker run -p 8020:8020 social-safety-amr-sam3-service` started the SAM 3 service in CPU mode, but `/health` returned degraded because `facebook/sam3` is gated and no authorized `HF_TOKEN` was available.
- `SAM3_SERVICE_URL=http://localhost:8020` dataset-service returned real SocialNav-SUB scenario `101_Spot_1_155` metadata from Hugging Face frames and reported the SAM 3 gated-model error through `/analysis` metadata.

Not yet complete:

- Successful real SAM 3 inference. The service adapter is present, but the current local run is blocked by gated Hugging Face access for `facebook/sam3`.
- GPU SAM 3 runtime through Docker Compose on this machine. Host `nvidia-smi` detects the GTX 1650, but Docker currently exposes only `runc` runtimes and `cuda-full` startup fails with `could not select device driver "" with capabilities: [[gpu]]`.
- Real RoboPoint grounding integration.
- Real VQA model inference.
- Real person/vehicle segmentation on arbitrary SocialNav-SUB or local MP4 videos.
- Stable multi-object Track IDs from SAM 3 official video tracking.
- Vehicle-specific detector initialization for bicycle, motorcycle, car, bus, and truck.
- Metric BEV from depth, intrinsics, extrinsics, ground plane, pose, or odometry.
- True m/s speed estimation; RGB-only output remains approximate normalized speed.
- Scene-change detection and automatic road re-grounding.
- Background inference worker process and bounded request-dropping queue for real model runtimes.
- Demo MP4 with real SAM 3/RoboPoint/VQA overlays.
- Formal SocialNav-SUB experiment metrics and overlay MP4 output.
- ROS 2 Humble adapter runtime.
- Jetson build validation.

## Formal Experiment Guardrails

- Mock or fixture results are not valid formal experiment outputs.
- If Hugging Face model or dataset access is blocked, record the blocking reason and do not mark formal smoke tests as passed.
- No fabricated metrics are permitted. Metrics files must be generated from actual predictions, decisions, and ground truth or explicitly marked unavailable.

## Known Local Environment

- Workspace Python default is 3.13; `/usr/bin/python3.10` is available and is used by `make setup`.
- Host GPU detected on 2026-06-29: NVIDIA GeForce GTX 1650 with Max-Q Design, 4096 MiB, driver 575.64.03.
- Docker NVIDIA runtime is not installed/configured on this machine; `docker info` reports only `runc`/containerd runtimes.
- SAM 3 model access requires an authorized `HF_TOKEN` for `facebook/sam3`; without it, the service stays degraded and no SAM 3 segmentation is produced.
