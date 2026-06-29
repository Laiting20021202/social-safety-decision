from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.frame_sources.settings import DatasetSettings
from services.dataset_service.app import build_default_source


def main() -> int:
    parser = argparse.ArgumentParser(description="List or lazily download SocialNav-SUB frames.")
    parser.add_argument("--list-only", action="store_true", help="Only list scenarios.")
    parser.add_argument("--scenario", default=None, help="Scenario ID to download.")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames to materialize.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    settings = DatasetSettings()
    source = build_default_source(settings)
    scenarios = source.list_scenarios()
    summary: dict[str, object] = {
        "dataset": source.dataset_info().model_dump(mode="json"),
        "scenario_count": len(scenarios),
        "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
    }

    if not args.list_only and scenarios:
        scenario_id = args.scenario or scenarios[0].scenario_id
        scenario = source.get_scenario(scenario_id)
        downloaded = []
        for index in range(min(args.max_frames, scenario.frame_count)):
            downloaded.append(str(source.get_frame_image_path(scenario_id, index)))
        summary["downloaded"] = {"scenario_id": scenario_id, "frames": downloaded}

    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
