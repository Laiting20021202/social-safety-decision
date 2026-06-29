# Implementation Plan

## Phase 0

Completed for an empty workspace:

1. Audit existing repository content.
2. Record reuse analysis, dependency risks, and license risks.
3. Create initial architecture, implementation plan, risk register, third-party record, and status file.

## Phase 1

Implemented target:

1. Shared Pydantic models.
2. `FrameSource` interface.
3. SocialNav-SUB Hugging Face/local-mirror adapter.
4. Playback manager and dataset-service API.
5. Basic React GUI with scenario selection, playback controls, image display, overlay toggles, and manual polygon editor.
6. Zone save/load.
7. Docker `dataset-demo` profile.
8. Unit and integration tests.

## Phase 1.5

Implemented target:

1. Cached MP4 generation for image-sequence scenarios.
2. HTML5 video playback with timestamp-based analysis synchronization.
3. Unified `AnalysisPacket` API.
4. Road / Path Calibration fallback replacing the primary manual danger-zone workflow.
5. Geometry helpers for velocity, direction, BEV projection, and dynamic risk zones.
6. Approximate RGB-only BEV and track list GUI.
7. Unit and integration tests for the video/analysis path.

## Next Phases

- Phase 2: strengthen geometry baseline, metrics, and overlay rendering.
- Phase 3: integrate official SAM 3 tracking in isolated service.
- Phase 4: integrate official RoboPoint and SAM 3 zone masking.
- Phase 5: integrate temporal VQA providers with strict JSON validation.
- Phase 6: complete deterministic safety fusion and resume logic.
- Phase 7: expand GUI inspectors, timeline, and experiment dashboard.
- Phase 8: validate CUDA, CPU, Jetson, and ROS 2 Docker profiles.
- Phase 9: run full ablation experiments without fabricated metrics.
