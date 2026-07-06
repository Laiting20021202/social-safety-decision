# E2E Validation

Formal E2E is only valid when all required model checkpoints are downloaded and
loaded on CUDA. Unit tests, Docker build success, and Python imports are not
formal inference success.

## Required Command

```bash
RUN_FORMAL_GPU_TESTS=1 make e2e-demo
```

## Required Evidence

The run must produce:

- `outputs/e2e_demo/screenshots/`
- `outputs/e2e_demo/browser_logs/`
- `outputs/e2e_demo/service_logs/`
- `outputs/e2e_demo/demo_overlay.mp4`
- `outputs/e2e_demo/analysis.jsonl`
- `outputs/e2e_demo/manifest.json`
- `outputs/e2e_demo/report.md`

## Acceptance Criteria

- Docker sees the RTX 4060-class GPU.
- `torch.cuda.is_available()` is true inside the SAM 3 container.
- `facebook/sam3` checkpoint loads.
- `facebook/sam3.1` checkpoint loads.
- A real SocialNav-SUB scenario is converted to MP4.
- SAM 3 produces true road/person masks.
- SAM 3.1 tracks at least 30 frames.
- At least one person keeps the same official object ID for 10 or more valid frames.
- Masks are not bounding-box rectangles.
- Motion, direction, speed, and dynamic risk zones are generated from track history.
- At least one formal VQA result is produced or a genuine runtime blocker is recorded.
- RoboPoint succeeds or a genuine hardware/model-access blocker is recorded.
- GUI video playback remains smooth and timestamp-synchronized.
- Browser console has no unhandled errors.

## Current Status

Blocked on 2026-06-29:

- `facebook/sam3` gated checkpoint access is not granted.
- `facebook/sam3.1` gated checkpoint access is not granted.
- Therefore no formal E2E run has been completed yet.
