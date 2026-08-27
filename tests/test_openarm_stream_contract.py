from __future__ import annotations

import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_koch_stream.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _parameter_array(parameter: str) -> list[str]:
    match = re.search(
        rf'-p\s+{re.escape(parameter)}:="\[([^\"]*)\]"',
        _runner_text(),
    )
    assert match is not None, f"missing quoted ROS array parameter: {parameter}"
    return [value.strip() for value in match.group(1).split(",") if value.strip()]


def test_openarm_stream_runner_has_valid_shell_and_paired_self_filter_geometry() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    frames = _parameter_array("self_filter.link_frames")
    radii = [float(value) for value in _parameter_array("self_filter.link_radii_m")]
    assert len(frames) == len(radii)
    assert len(frames) == len(set(frames))
    assert all(math.isfinite(radius) and radius > 0.0 for radius in radii)

    urdf = (
        ROOT
        / "third_party/openarm_description/generated/openarm_v10_bimanual.urdf"
    )
    robot = ET.parse(urdf).getroot()
    collision_links = {
        link.attrib["name"]
        for link in robot.findall("link")
        if link.find("collision") is not None
    }
    # The central pedestal is static and its one-origin sphere would erase most
    # of the workcell. Every moving collision link must otherwise be covered.
    expected_frames = collision_links - {"openarm_body_link0"}
    assert set(frames) == expected_frames
    assert not any(frame.endswith("_tcp") for frame in frames)


def test_openarm_hand_candidates_fail_closed_without_rgb_hand_semantics() -> None:
    runner = _runner_text()
    node_source = (
        ROOT / "realtime_safety/edgetam_tracker/tracked_obstacle_node.py"
    ).read_text(encoding="utf-8")

    assert "-p self_filter.fail_closed:=true" in runner
    assert "-p hand_candidate.enabled:=true" in runner
    assert "-p hand_semantics.enabled:=true" in runner
    assert re.search(
        r"fail_closed_on_detector_unavailable\s*=\s*True",
        node_source,
    )


def test_native_coco_weights_are_labeled_as_objects_not_hand_yolo() -> None:
    profile = yaml.safe_load(
        (ROOT / "configs/koch_lan.yaml").read_text(encoding="utf-8")
    )
    models = profile["segmentation"]["model_options"]
    assert models
    assert all(model.startswith("yolo") and model.endswith("-seg.pt") for model in models)
    assert profile["segmentation"]["hand_only"] is True

    dashboard_source = (
        ROOT / "realtime_safety/gui/dashboard.py"
    ).read_text(encoding="utf-8")
    assert '"yolo": "Legacy COCO YOLO / MediaPipe Hand Gate + RGB-D"' in dashboard_source
    assert '"COCO YOLO object checkpoint (not hand model)"' in dashboard_source
    assert "These 80-class COCO weights are **not a human-hand model**" in dashboard_source
