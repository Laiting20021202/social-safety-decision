from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvaluationRecorder:
    output_root: Path
    scenario: str
    seed: int
    run_dir: Path = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "correctly_sorted_cubes": 0,
            "dropped_cubes": 0,
            "collisions": 0,
            "false_stops": 0,
            "physical_grasp_successes": 0,
            "magnetic_fallbacks": 0,
        }
    )
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stamp = os.environ.get("OPENARM_RUN_ID")
        if stamp is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        if not stamp.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError("OPENARM_RUN_ID may contain only letters, digits, dash, underscore, dot")
        self.run_dir = self.output_root / stamp
        for name in ("config_snapshot", "screenshots", "rosbag"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def snapshot_configs(self, config_dir: Path) -> None:
        for source in sorted(config_dir.glob("*.yaml")):
            shutil.copy2(source, self.run_dir / "config_snapshot" / source.name)

    def event(self, timestamp_sec: float, event: str, **details: Any) -> None:
        self.events.append({"timestamp_sec": timestamp_sec, "event": event, **details})
        self.timings.setdefault(event, timestamp_sec)

    def joint_sample(
        self,
        timestamp_sec: float,
        joint_names: list[str],
        position: list[float],
        velocity: list[float],
        effort: list[float] | None = None,
    ) -> None:
        effort_values = effort or [0.0] * len(joint_names)
        for index, name in enumerate(joint_names):
            self.trajectory.append(
                {
                    "timestamp_sec": timestamp_sec,
                    "joint": name,
                    "position": position[index],
                    "velocity": velocity[index],
                    "effort": effort_values[index],
                }
            )

    def finalize(self, task_complete: bool, min_distance_m: float | None = None) -> Path:
        self._add_derivatives()
        metrics: dict[str, Any] = {
            "scenario": self.scenario,
            "scenario_seed": self.seed,
            "task_complete": task_complete,
            "physical_grasp_success": self.counters[
                "physical_grasp_successes"
            ] > 0,
            "magnetic_assisted_task_success": bool(
                task_complete and self.counters["magnetic_fallbacks"] > 0
            ),
            "ground_truth_min_distance_m": min_distance_m,
            **self.counters,
            "perception_latency_sec": self._duration("hand_entered", "perception_output"),
            "reaction_latency_sec": self._duration("hand_entered", "safety_command"),
            "stop_latency_sec": self._duration("safety_command", "robot_stopped"),
            "replan_latency_sec": self._duration("replan_requested", "replan_complete"),
            "recovery_time_sec": self._duration("hand_withdrawn", "motion_resumed"),
        }
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_csv(self.run_dir / "events.csv", self.events)
        self._write_csv(self.run_dir / "trajectory.csv", self.trajectory)
        return self.run_dir

    def _duration(self, start: str, end: str) -> float | None:
        if start not in self.timings or end not in self.timings:
            return None
        return max(0.0, self.timings[end] - self.timings[start])

    def _add_derivatives(self) -> None:
        by_joint: dict[str, list[dict[str, Any]]] = {}
        for row in self.trajectory:
            by_joint.setdefault(str(row["joint"]), []).append(row)
        for samples in by_joint.values():
            samples.sort(key=lambda row: float(row["timestamp_sec"]))
            times = np.asarray([row["timestamp_sec"] for row in samples], dtype=float)
            velocities = np.asarray([row["velocity"] for row in samples], dtype=float)
            if len(samples) < 2 or np.any(np.diff(times) <= 0.0):
                acceleration = np.zeros_like(velocities)
                jerk = np.zeros_like(velocities)
            else:
                acceleration = np.gradient(velocities, times)
                jerk = np.gradient(acceleration, times)
            for row, acc, joint_jerk in zip(samples, acceleration, jerk, strict=True):
                row["acceleration"] = float(acc)
                row["jerk"] = float(joint_jerk)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
