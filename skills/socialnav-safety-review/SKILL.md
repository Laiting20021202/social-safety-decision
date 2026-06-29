---
name: socialnav-safety-review
description: Review social-navigation AMR safety logic for unsafe continue decisions, hard-rule priority, VQA direct-control violations, time-to-zone validation, safe-resume criteria, fail-safe behavior, data leakage, and mock-result contamination.
---

# SocialNav Safety Review

Use this workflow when reviewing safety logic, VQA integration, geometry prediction, controller state transitions, experiment metrics, or formal reports.

## Workflow

1. Check control authority:
   - VQA may recommend but must not directly control AMR motion
   - no model may publish arbitrary `/cmd_vel`
2. Check priority order:
   - hard safety rules
   - tracking and geometry
   - time-to-zone prediction
   - VQA semantic reasoning
   - navigation recommendation
3. Check fail-safe behavior:
   - timeout
   - invalid JSON
   - low confidence
   - missing zone
   - missing timestamp
   - tracking failure
   - contradictory model outputs
4. Check deterministic rules:
   - person-zone intersection pauses
   - critical time-to-zone pauses
   - warning time-to-zone slows or yields
   - low-confidence near-zone pauses or requests review
5. Check safe resume:
   - zone clear
   - no approaching person
   - minimum clear duration
   - valid tracking
   - geometry indicates safe
   - VQA safe-resume check if enabled
6. Check experiment integrity:
   - mock results excluded from formal runs
   - no fabricated metrics
   - critical false negatives and unsafe continue are reported

## Required Checks

- unsafe continue rate
- future zone-entry recall
- critical false-negative rate
- safe-resume accuracy
- invalid JSON rate
- timeout rate

## Failure Conditions

- uncertainty produces `continue`
- VQA override weakens a hard rule
- safe resume occurs immediately after zone exit without clearance
- formal metrics include mock provider outputs
- missing timestamp is silently treated as fixed FPS
- critical false negatives are hidden by overall accuracy

## Output Format

Lead with findings by severity. Include file and line references when reviewing code. If no issues are found, state residual test gaps.
