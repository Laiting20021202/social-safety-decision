from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from packages.common_models import ExperimentConfig
from packages.frame_sources.settings import DatasetSettings
from services.dataset_service.app import build_default_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 dataset playback smoke checks.")
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--formal", default="false", choices=["true", "false"])
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    formal = args.formal == "true"
    if formal:
        print(
            "Formal smoke is blocked in Phase 1 because SAM 3, RoboPoint, and VQA real inference "
            "are not integrated yet."
        )
        return 2

    settings = DatasetSettings()
    source = build_default_source(settings)
    scenarios = source.list_scenarios()[: args.scenarios]
    run_id = datetime.now(timezone.utc).strftime("phase1-smoke-%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        run_id=run_id,
        dataset_id=source.dataset_info().dataset_id,
        dataset_revision=source.dataset_info().revision,
        scenarios=[scenario.scenario_id for scenario in scenarios],
        formal=False,
    )
    (output_dir / "config.yaml").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "dataset_info.json").write_text(
        source.dataset_info().model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "formal": False,
                "metrics_available": False,
                "reason": "Phase 1 smoke validates playback only; no formal model predictions.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# Phase 1 Smoke Report\n\n"
        "Dataset playback smoke artifacts were created. Formal SAM 3, RoboPoint, and VQA "
        "checks are not part of Phase 1 and are not marked passed.\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "scenarios": config.scenarios,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
