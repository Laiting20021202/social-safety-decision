# GUI User Guide

Start the dataset service and web app, then open the browser UI.

The Phase 1.5 GUI supports:

- selecting a SocialNav-SUB scenario
- continuous HTML5 video playback from cached MP4 assets
- play, pause, restart, seek, playback speed, prediction horizon, and VQA update interval
- timestamp-synchronized analysis overlays using `video.currentTime`
- SAM 3 road segmentation overlay when the SAM 3 service returns a real mask
- explicit fallback path editor controls; fallback paths are not formal SAM 3 output
- person/vehicle track overlays when real tracker output or marked fixture output is available
- direction arrows, approximate speed, and risk state labels
- BEV Safety Map with robot corridor, SAM 3/fallback path when available, and red dynamic risk zones
- track list with direction, speed, path relation, VQA status, and risk
- collapsed debug JSON for the latest `AnalysisPacket`

SAM 3 is integrated as a separate service adapter. On this machine it currently returns degraded until `facebook/sam3` can be downloaded with an authorized `HF_TOKEN`, and Docker GPU execution also requires NVIDIA Container Toolkit. VQA, RoboPoint, and formal decision output remain unavailable until their services are integrated. Fixture-only segmentation is marked as non-formal output.
