import csv
import json

import pytest

from openarm_sim.metrics import EvaluationRecorder


def test_metrics_artifact_tree_and_derivatives(tmp_path) -> None:
    recorder = EvaluationRecorder(tmp_path, "no_obstacle", 7)
    recorder.event(0.1, "hand_entered")
    recorder.event(0.2, "perception_output")
    recorder.event(0.3, "safety_command")
    recorder.event(0.4, "robot_stopped")
    for index in range(4):
        recorder.joint_sample(index * 0.1, ["joint"], [index * 0.01], [index * 0.1])
    output = recorder.finalize(task_complete=True, min_distance_m=0.21)
    assert (output / "metrics.json").is_file()
    assert (output / "trajectory.csv").is_file()
    assert (output / "events.csv").is_file()
    assert (output / "config_snapshot").is_dir()
    assert (output / "screenshots").is_dir()
    assert (output / "rosbag").is_dir()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["perception_latency_sec"] == 0.1
    assert metrics["reaction_latency_sec"] == pytest.approx(0.2)
    with (output / "trajectory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {"acceleration", "jerk"}.issubset(rows[0])
