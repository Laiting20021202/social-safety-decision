#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local SCAND sample manifest")
    parser.add_argument("--sample", default="data/scand_sample", help="Sample directory")
    parser.add_argument("--limit", type=int, default=5, help="Print first N records")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = Path(args.sample)
    manifest = sample / "manifest.jsonl"
    image_dir = sample / "images"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    print(f"sample: {sample}")
    print(f"images: {len(list(image_dir.glob('*')))}")
    print(f"manifest records: {len(records)}")
    for record in records[: args.limit]:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

