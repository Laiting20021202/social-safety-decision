# CHANGELOG

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
