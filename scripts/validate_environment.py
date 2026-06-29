from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return str(output).strip()
    except subprocess.CalledProcessError as exc:
        return str(exc.output).strip()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    git_commit = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if git_commit and git_commit.startswith("fatal:"):
        git_commit = None
    status = {
        "os": platform.platform(),
        "python_version": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "docker": _run(["docker", "--version"]),
        "docker_compose": _run(["docker", "compose", "version"]),
        "node": _run(["node", "--version"]),
        "npm": _run(["npm", "--version"]),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": git_commit,
        "app_mode": os.getenv("APP_MODE", "dataset_playback"),
    }
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
