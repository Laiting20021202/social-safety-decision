from __future__ import annotations

import json
import re
from pathlib import Path

from packages.common_models import ZoneDefinition


class ZoneStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, scenario_id: str) -> ZoneDefinition | None:
        path = self._path(dataset_name, scenario_id)
        if not path.exists():
            return None
        return ZoneDefinition.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, dataset_name: str, zone: ZoneDefinition) -> Path:
        path = self._path(dataset_name, zone.scenario_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(zone.model_dump_json(indent=2), encoding="utf-8")
        return path

    def delete(self, dataset_name: str, scenario_id: str) -> bool:
        path = self._path(dataset_name, scenario_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def export_dict(self, dataset_name: str, scenario_id: str) -> dict[str, object] | None:
        zone = self.load(dataset_name, scenario_id)
        if zone is None:
            return None
        data: dict[str, object] = json.loads(zone.model_dump_json())
        return data

    def _path(self, dataset_name: str, scenario_id: str) -> Path:
        return self.root / _safe_name(dataset_name) / f"{_safe_name(scenario_id)}.json"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)
