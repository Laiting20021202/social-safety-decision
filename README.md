# social-safety-amr

AMR social-navigation safety analysis workbench for smooth real-image dataset video playback, SAM 3 segmentation integration, track motion estimation, BEV safety-map rendering, temporal VQA integration points, and deterministic safety decisions.

The current implementation is Phase 1.5 focused:

- Offline dataset playback with `michaelmunje/SocialNav-SUB` frame sequences from Hugging Face.
- Cached MP4 generation for image-sequence scenarios and HTML5 video playback in the GUI.
- Timestamp-synchronized `AnalysisPacket` metadata API for SAM 3 road masks, person/vehicle tracks, motion estimates, dynamic risk zones, robot corridor, and analysis delay status.
- Unified `FrameSource` and Pydantic v2 data models.
- FastAPI `dataset-service` with playback REST API, video endpoints, analysis endpoints, WebSocket playback metadata, SAM 3 client integration, and optional fallback road persistence.
- FastAPI `sam3-service` container image for SAM 3 image segmentation. If the model cannot be loaded, the service returns a degraded health status and the analysis API reports the exact blocking reason.
- React/Vite TypeScript GUI with scenario selection, HTML5 video controls, SAM 3 road/agent overlays, agent direction/speed overlay, BEV Safety Map, red dynamic risk zones, and track list.
- Docker Compose `dataset-demo` profile.
- Unit and integration tests for frame schemas, playback, cached MP4, road schemas, VQA JSON parsing, velocity estimation, direction fusion, dynamic risk zones, BEV transforms, stale analysis rejection, and API behavior.

RoboPoint, VQA, ROS 2, and full experiment execution remain scaffolded as independent services for later phases. Mock or fixture-only output is not used as formal experiment output. The local test fixture uses explicitly marked `fixture_color_segmentation` so the video/BEV data flow can be tested without claiming real SAM 3 or RoboPoint inference.

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

SAM 3 service:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full build sam3-service
HF_TOKEN=... docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full up sam3-service dataset-service web-gui
```

`facebook/sam3` is a gated Hugging Face model. Without a token that has access to that model, `/health` returns degraded and `/analysis` reports SAM 3 unavailable instead of fabricating segmentation.

## GUI Workflow

The browser GUI is organized into three areas:

- Left: dataset, scenario, load, play/pause, seek, playback speed, prediction horizon, VQA interval, overlay toggles, and explicit SAM 3 fallback editor controls.
- Center: continuous HTML5 video overlay and BEV Safety Map.
- Right: track list with direction, approximate speed, path relation, VQA status, risk level, and a collapsed debug JSON drawer.

When the dataset provides only image frames, `/scenarios/{scenario_id}/video-info` builds a cached MP4 with `ffmpeg`, and `/scenarios/{scenario_id}/video` serves that MP4. Analysis metadata is fetched separately from `/scenarios/{scenario_id}/analysis` using `video.currentTime`; model inference must not control video playback speed.

SAM 3 road segmentation is the primary path source. If SAM 3 road grounding is unavailable, the GUI shows `Road segmentation unavailable`, does not draw an auto-generated road polygon, and allows a saved fallback path only after the fallback editor is explicitly unlocked. The fallback is labeled `manual_fallback` and is not a formal SAM 3 result.

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
