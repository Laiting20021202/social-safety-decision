from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    if value := os.environ.get("OPENARM_SIM_ROOT"):
        return Path(value).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config").is_dir():
        return source_root
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("openarm_sim_bringup"))
    except (ImportError, LookupError):
        return source_root


PROJECT_ROOT = _project_root()


class ConfigError(ValueError):
    """Raised when a project configuration is missing or internally inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} missing required keys: {', '.join(missing)}")
