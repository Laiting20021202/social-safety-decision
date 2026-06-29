---
name: robotics-reproducible-experiment
description: Build, audit, and validate reproducible robotics experiments for social-safety AMR work, including dataset/model revision pinning, environment capture, artifact validation, metrics integrity, failure-case generation, report generation, and fabricated-result prevention.
---

# Robotics Reproducible Experiment

Use this workflow when creating or reviewing experiment runs.

## Workflow

1. Pin inputs before running:
   - dataset repository URL and revision
   - model repository URL and revision
   - code git commit
   - container image digest
   - experiment config
2. Capture runtime environment:
   - OS, CPU, GPU, CUDA, driver, RAM, VRAM
   - container image and digest
   - Python, ROS 2, and package versions
   - hardware target and hostname
3. Validate config:
   - reject missing prediction horizon, time steps, sampling interval, or zone source
   - reject formal runs that use mock providers
   - reject unpinned datasets or model revisions
4. Write artifacts under `outputs/<run_id>/`:
   - `config.yaml`
   - `environment.json`
   - `git_info.json`
   - `container_info.json`
   - `dataset_info.json`
   - `model_info.json`
   - `predictions.jsonl`
   - `tracks.jsonl`
   - `vqa_requests.jsonl`
   - `decisions.jsonl`
   - `metrics.json`
   - `report.md`
5. Validate metrics:
   - compute metrics only from actual outputs and ground truth
   - mark unavailable metrics explicitly
   - never fill missing values with invented numbers
6. Generate failure cases for unsafe continue, missed zone entry, incorrect resume, invalid JSON, timeout, tracking loss, wrong zone, and system errors.
7. Write a report that distinguishes passed, failed, blocked, and unavailable checks.

## Required Checks

- `future_zone_entry_recall`
- `critical_false_negative_rate`
- `unsafe_continue_rate`
- `safe_resume_accuracy`
- end-to-end latency
- invalid JSON rate
- timeout rate

## Failure Conditions

- mock provider used in a formal run
- model or dataset revision is not recorded
- metrics were not derived from saved predictions and decisions
- formal run has missing artifacts without a blocking reason
- report claims success while required model or dataset access is blocked

## Output Format

Summarize:

- run ID
- config path
- artifact completeness
- metric availability
- blocking reasons
- safety-critical failures
