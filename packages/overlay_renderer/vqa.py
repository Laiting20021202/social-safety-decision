from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from packages.common_models import VQADirectionEstimate


def parse_vqa_direction_json(raw_response: str, track_id: int) -> VQADirectionEstimate:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return VQADirectionEstimate(
            track_id=track_id,
            parse_valid=False,
            reason=f"JSON parse error: {exc.msg}",
        )
    if not isinstance(data, dict):
        return VQADirectionEstimate(
            track_id=track_id,
            parse_valid=False,
            reason="VQA response must be a JSON object.",
        )
    data.setdefault("track_id", track_id)
    try:
        parsed = VQADirectionEstimate.model_validate(data)
    except ValidationError as exc:
        return VQADirectionEstimate(
            track_id=track_id,
            parse_valid=False,
            reason=f"Schema validation error: {exc.errors()[0]['msg']}",
        )
    return parsed.model_copy(update={"parse_valid": True})


def vqa_direction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "track_id",
            "motion_state",
            "direction_label",
            "path_relation",
            "confidence",
            "reason",
        ],
        "properties": {
            "track_id": {"type": "integer"},
            "motion_state": {"enum": ["moving", "stationary", "uncertain"]},
            "direction_label": {
                "enum": [
                    "toward_camera",
                    "away_from_camera",
                    "left",
                    "right",
                    "forward_left",
                    "forward_right",
                    "backward_left",
                    "backward_right",
                    "stationary",
                    "uncertain",
                ]
            },
            "path_relation": {
                "enum": [
                    "along_path",
                    "crossing_path",
                    "entering_path",
                    "leaving_path",
                    "parallel_to_path",
                    "uncertain",
                ]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }
