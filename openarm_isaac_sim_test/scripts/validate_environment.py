#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))


def _command(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def collect() -> dict[str, Any]:
    isaac_root = Path(os.environ.get("ISAAC_SIM_ROOT", "/home/david/isaacsim"))
    lab_candidates = [
        Path(os.environ["ISAACLAB_ROOT"]) if "ISAACLAB_ROOT" in os.environ else None,
        PROJECT_ROOT.parent / "IsaacLab",
    ]
    lab_root = next((path for path in lab_candidates if path and path.is_dir()), None)
    openarm_candidates = [
        Path(os.environ["OPENARM_DESCRIPTION_ROOT"])
        if "OPENARM_DESCRIPTION_ROOT" in os.environ
        else None,
        PROJECT_ROOT / "ros2_ws/src/external/openarm_description",
        PROJECT_ROOT.parent / "third_party/openarm_description",
    ]
    description_root = next(
        (path for path in openarm_candidates if path and path.is_dir()), None
    )
    return {
        "project_root": str(PROJECT_ROOT),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": _command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "cuda_compatibility": _command("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"),
        "ros_distro": os.environ.get("ROS_DISTRO") or ("humble" if Path("/opt/ros/humble").is_dir() else None),
        "ros2": shutil.which("ros2"),
        "moveit_system_package": _command("dpkg-query", "-W", "-f=${Status} ${Version}", "ros-humble-moveit"),
        "moveit_overlay": str(Path(os.environ["MOVEIT_OVERLAY"]))
        if "MOVEIT_OVERLAY" in os.environ and Path(os.environ["MOVEIT_OVERLAY"]).is_dir()
        else None,
        "isaac_sim_root": str(isaac_root) if isaac_root.is_dir() else None,
        "isaac_sim_version": (isaac_root / "VERSION").read_text(encoding="utf-8").strip()
        if (isaac_root / "VERSION").is_file()
        else None,
        "isaac_lab_root": str(lab_root) if lab_root else None,
        "isaac_lab_git": _command("git", "-C", str(lab_root), "describe", "--always", "--dirty")
        if lab_root
        else None,
        "openarm_description_root": str(description_root) if description_root else None,
        "openarm_description_git": _command(
            "git", "-C", str(description_root), "rev-parse", "HEAD"
        )
        if description_root
        else None,
        "required_tools": {
            name: shutil.which(name) for name in ("colcon", "ros2", "xacro", "vcs")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only OpenArm environment audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = collect()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    required = (report["isaac_sim_root"], report["openarm_description_root"])
    return 0 if all(required) else 2


if __name__ == "__main__":
    raise SystemExit(main())
