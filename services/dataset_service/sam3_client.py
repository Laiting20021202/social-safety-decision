from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


class Sam3Client:
    def __init__(self, base_url: str | None = None, timeout_sec: float = 20.0) -> None:
        self.base_url = (base_url or os.getenv("SAM3_SERVICE_URL") or "").rstrip("/")
        self.timeout_sec = timeout_sec

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def segment_image(self, image_path: Path, prompts: list[str]) -> dict[str, Any] | None:
        if not self.configured:
            return None
        with image_path.open("rb") as handle:
            files = {"image": (image_path.name, handle, _content_type(image_path))}
            data = {"prompts": ",".join(prompts)}
            try:
                response = httpx.post(
                    f"{self.base_url}/segment-image",
                    files=files,
                    data=data,
                    timeout=self.timeout_sec,
                )
            except httpx.HTTPError as exc:
                return {
                    "source": "sam3_unavailable",
                    "detections": [],
                    "message": f"SAM3 request failed: {exc}",
                }
        if response.status_code >= 400:
            return {
                "source": "sam3_unavailable",
                "detections": [],
                "message": _response_message(response),
            }
        result = response.json()
        return result if isinstance(result, dict) else None


def _content_type(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _response_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return response.text
