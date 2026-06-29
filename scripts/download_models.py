from __future__ import annotations

import json


def main() -> int:
    status = {
        "sam3": {
            "repository": "https://github.com/facebookresearch/sam3",
            "status": "not_downloaded",
            "blocking_reason": "Phase 3 integration not executed yet.",
        },
        "robopoint": {
            "repository": "https://github.com/wentaoyuan/RoboPoint",
            "status": "not_downloaded",
            "blocking_reason": "Phase 4 integration not executed yet.",
        },
        "qwen": {
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "status": "not_downloaded",
            "blocking_reason": "Phase 5 provider integration not executed yet.",
        },
        "smolvlm": {
            "model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "status": "not_downloaded",
            "blocking_reason": "Phase 5 provider integration not executed yet.",
        },
    }
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
