from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class RobotAsset:
    kind: str
    path: Path
    source_root: Path


def resolve_robot_asset(robot_config: dict[str, Any]) -> RobotAsset:
    asset = robot_config["asset"]
    usd_value = os.environ.get(asset["usd_env"], "").strip()
    if usd_value:
        usd_path = Path(usd_value).expanduser().resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(f"OPENARM_USD_PATH does not exist: {usd_path}")
        return RobotAsset("usd", usd_path, usd_path.parent)

    root_value = os.environ.get(asset["description_root_env"], "").strip()
    candidates = ([root_value] if root_value else []) + list(asset["description_root_candidates"])
    source_root = next(
        (Path(candidate).expanduser().resolve() for candidate in candidates if Path(candidate).is_dir()),
        None,
    )
    if source_root is None:
        raise FileNotFoundError(
            "OpenArm assets not found. Set OPENARM_DESCRIPTION_ROOT or OPENARM_USD_PATH; "
            f"official source: {asset['source_repository']}"
        )

    generated = source_root / asset["generated_urdf"]
    if not generated.is_file():
        generated = _generate_urdf(source_root)
    cache_dir = PROJECT_ROOT / "assets" / "openarm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_urdf = cache_dir / "openarm_v10_bimanual.resolved.urdf"
    text = generated.read_text(encoding="utf-8")
    package_prefix = source_root.as_uri().rstrip("/") + "/"
    text = text.replace("package://openarm_description/", package_prefix)
    resolved_urdf.write_text(text, encoding="utf-8")
    metadata = {
        "source_root": str(source_root),
        "source_repository": asset["source_repository"],
        "expected_revision": asset["pinned_revision"],
        "source_urdf": str(generated),
    }
    (cache_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return RobotAsset("urdf", resolved_urdf, source_root)


def _generate_urdf(source_root: Path) -> Path:
    xacro_path = source_root / "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
    if not xacro_path.is_file():
        raise FileNotFoundError(f"OpenArm xacro not found: {xacro_path}")
    output_dir = PROJECT_ROOT / "assets" / "openarm_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "openarm_v10_bimanual.generated.urdf"
    command = [
        "xacro",
        str(xacro_path),
        "bimanual:=true",
        "hand:=true",
        "ee_type:=parallel_link",
        "use_fake_hardware:=true",
        "can_fd:=true",
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=source_root, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to generate OpenArm URDF with xacro:\n"
            + completed.stdout
            + completed.stderr
        )
    return output_path

