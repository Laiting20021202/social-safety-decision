#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationCase:
    name: str
    scenario: str
    mode: str
    environment: dict[str, str]


CASES = (
    ValidationCase("baseline", "no_obstacle", "ground_truth", {}),
    ValidationCase("static_obstacle", "static_blocking", "ground_truth", {}),
    ValidationCase("moving_crossing", "right_side_sweep", "ground_truth", {}),
    ValidationCase("sudden_intrusion", "sudden_intrusion", "ground_truth", {}),
    ValidationCase("withdraw_recover", "intrude_and_withdraw", "ground_truth", {}),
    ValidationCase("fully_blocked", "fully_blocked", "ground_truth", {}),
    ValidationCase(
        "perception_timeout",
        "static_blocking",
        "perception",
        {"OPENARM_CAMERA_FRAME_DROP_PROBABILITY": "1.0"},
    ),
    ValidationCase(
        "depth_frame_drop",
        "right_side_sweep",
        "perception",
        {"OPENARM_CAMERA_FRAME_DROP_PROBABILITY": "0.35"},
    ),
    ValidationCase("repeated_intrusion", "repeated_intrusion", "ground_truth", {}),
)


def _command(case: ValidationCase, output_root: Path) -> list[str]:
    launch = (
        "ground_truth_validation.launch.py"
        if case.mode == "ground_truth"
        else "perception_validation.launch.py"
    )
    return [
        "ros2",
        "launch",
        "openarm_sim_bringup",
        launch,
        f"scenario:={case.scenario}",
        "headless:=true",
        "use_rviz:=false",
        f"output_root:={output_root}",
    ]


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)


def run_case(
    case: ValidationCase,
    output_root: Path,
    timeout_sec: float,
    record_rosbag: bool,
    dry_run: bool,
) -> bool:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{timestamp}_{case.name}"
    run_dir = output_root / run_id
    command = _command(case, output_root)
    environment = os.environ.copy()
    environment.update(case.environment)
    environment["OPENARM_RUN_ID"] = run_id
    environment["OPENARM_OUTPUT_ROOT"] = str(output_root.resolve())
    environment.setdefault("OPENARM_SIM_ROOT", str(PROJECT_ROOT))
    print(f"[{case.name}] {shlex.join(command)}")
    if dry_run:
        if case.environment:
            print(f"  env: {case.environment}")
        return True
    run_dir.mkdir(parents=True, exist_ok=False)
    bag: subprocess.Popen[bytes] | None = None
    launch: subprocess.Popen[bytes] | None = None
    try:
        if record_rosbag:
            bag_command = [
                "ros2",
                "bag",
                "record",
                "-o",
                str(run_dir / "rosbag"),
                "/clock",
                "/joint_states",
                "/openarm/events",
                "/openarm/safety/state",
                "/openarm/safety/velocity_scaling",
                "/rgbd/color/image_raw",
                "/rgbd/depth/image_raw",
                "/rgbd/points",
                "/sim/ground_truth/hand_pose",
                "/sim/ground_truth/min_distance",
            ]
            bag = subprocess.Popen(bag_command, env=environment, start_new_session=True)
        launch = subprocess.Popen(command, env=environment, start_new_session=True)
        deadline = time.monotonic() + timeout_sec
        metrics = run_dir / "metrics.json"
        while time.monotonic() < deadline:
            if launch.poll() is not None:
                break
            if metrics.is_file():
                return True
            time.sleep(0.5)
        return metrics.is_file()
    finally:
        _terminate(launch)
        _terminate(bag)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic OpenArm validation cases")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--mode", choices=("all", "ground_truth", "perception"), default="all")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--no-rosbag", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [
        case
        for case in CASES
        if (args.mode == "all" or case.mode == args.mode)
        and (not args.case or case.name in args.case)
    ]
    unknown = set(args.case) - {case.name for case in CASES}
    if unknown:
        parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
    outcomes = [
        run_case(
            case,
            args.output_root.resolve(),
            args.timeout_sec,
            not args.no_rosbag,
            args.dry_run,
        )
        for case in selected
    ]
    passed = sum(outcomes)
    print(f"VALIDATION_SUITE_RESULT passed={passed} total={len(outcomes)}")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
