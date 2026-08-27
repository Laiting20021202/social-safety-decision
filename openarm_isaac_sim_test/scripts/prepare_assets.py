#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))

from openarm_sim.assets import resolve_robot_asset
from openarm_sim.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the pinned official OpenArm asset")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    asset = resolve_robot_asset(load_yaml("config/openarm.yaml")["robot"])
    result = {"kind": asset.kind, "path": str(asset.path), "source": str(asset.source_root)}
    if args.print_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"OPENARM_ASSET_KIND={asset.kind}")
        print(f"OPENARM_ASSET_PATH={asset.path}")
        print(f"OPENARM_ASSET_SOURCE={asset.source_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
