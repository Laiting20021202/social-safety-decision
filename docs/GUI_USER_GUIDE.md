# GUI User Guide

Start the CUDA stack:

```bash
make doctor
make check-model-access
make download-models
make precompute-demo
make demo-cuda
```

Open:

```text
http://localhost:5173
```

## Workflow

1. Select the SocialNav-SUB dataset.
2. Select a scenario.
3. Click `Load`.
4. If a formal precompute already exists, press play.
5. If not, click `Re-run Analysis`.
6. Use the video timeline as the main time axis.
7. Inspect the video overlay, BEV Safety Map, Track List, and System Status.

## Status Rules

- `Precompute` shows whether background analysis is running, completed, or blocked.
- `SAM3 runtime` shows the real blocking reason, including gated Hugging Face access.
- `Formal output` must be `true` before results are treated as formal model output.
- Manual road fallback remains labeled as fallback and must not be interpreted as RoboPoint or SAM 3 output.

## Current Blocker

As of 2026-06-29, the GUI can operate, play cached video, and show status, but
formal SAM 3 / SAM 3.1 overlays are blocked until `facebook/sam3` and
`facebook/sam3.1` checkpoint downloads succeed with an authorized `HF_TOKEN`.
