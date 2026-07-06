from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """Raised when a YAML configuration is invalid or unreadable."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"Configuration file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a YAML mapping: {yaml_path}")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load default config and optionally merge a user config."""

    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_yaml(default_path)
    if path:
        config = deep_merge(config, load_yaml(path))
    return config


def load_classes(path: str | Path | None = None) -> dict[str, list[str]]:
    classes_path = Path(path) if path else PROJECT_ROOT / "configs" / "classes.yaml"
    data = load_yaml(classes_path)
    return {
        "walkable_classes": list(data.get("walkable_classes", [])),
        "optional_walkable_classes": list(data.get("optional_walkable_classes", [])),
        "blocked_classes": list(data.get("blocked_classes", [])),
    }


def load_detection_config(path: str | Path | None = None) -> dict[str, Any]:
    detection_path = Path(path) if path else PROJECT_ROOT / "configs" / "detection.yaml"
    return load_yaml(detection_path)


def resolve_path(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (Path(base) if base else PROJECT_ROOT) / resolved

