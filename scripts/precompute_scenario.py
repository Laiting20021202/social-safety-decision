from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return parsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute a SocialNav-SUB scenario.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--dataset", default="socialnav_sub")
    parser.add_argument("--dataset-service-url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    base_url = args.dataset_service_url.rstrip("/")
    out_dir = Path("precomputed") / args.dataset / args.scenario
    logs_dir = out_dir / "logs"
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "scenario_id": args.scenario,
        "created_at": created_at,
        "status": "started",
        "formal_model_output": False,
        "dataset_service_url": base_url,
        "outputs": {
            "manifest": str(out_dir / "manifest.json"),
            "logs": str(logs_dir),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    try:
        health = request_json(f"{base_url}/health")
        video_info = request_json(f"{base_url}/scenarios/{args.scenario}/video-info")
        response = request_json(
            f"{base_url}/scenarios/{args.scenario}/analysis/sam3-video",
            method="POST",
            payload={
                "prompt": args.prompt,
                "prompt_frame_index": 0,
                "max_frame_num_to_track": args.max_frames,
            },
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        manifest.update(
            {
                "status": "blocked",
                "blocking_reason": f"HTTP {exc.code}: {detail}",
                "formal_model_output": False,
            }
        )
        write_json(logs_dir / "error.json", {"error": manifest["blocking_reason"]})
        write_json(out_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001 - command report should preserve exact blocker
        manifest.update(
            {
                "status": "blocked",
                "blocking_reason": str(exc),
                "formal_model_output": False,
            }
        )
        write_json(logs_dir / "error.json", {"error": str(exc)})
        write_json(out_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    write_json(logs_dir / "health.json", health)
    write_json(logs_dir / "video_info.json", video_info)
    write_json(logs_dir / "sam3_video_response.json", response)

    ok = response.get("status") == "ok"
    manifest.update(
        {
            "status": "completed" if ok else "blocked",
            "formal_model_output": bool(ok),
            "blocking_reason": None if ok else response.get("message") or "SAM3 precompute failed.",
            "video_info": video_info,
            "sam3_video": response,
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
