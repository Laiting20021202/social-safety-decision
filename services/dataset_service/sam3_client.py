from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


class Sam3Client:
    def __init__(self, base_url: str | None = None, timeout_sec: float = 600.0) -> None:
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

    def start_video_session(
        self,
        resource_path: Path,
        session_id: str | None = None,
        scenario_id: str | None = None,
        analysis_fps: float = 5.0,
        max_objects: int = 6,
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None
        payload: dict[str, Any] = {
            "video_path": str(resource_path),
            "scenario_id": scenario_id,
            "analysis_fps": analysis_fps,
            "max_objects": max_objects,
        }
        if session_id:
            payload["session_id"] = session_id
        return self._post_json("/video/sessions", payload)

    def add_video_prompt(
        self,
        session_id: str,
        prompt: str,
        frame_index: int = 0,
        output_prob_thresh: float = 0.5,
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None
        return self._post_json(
            f"/video/sessions/{session_id}/prompts",
            {
                "prompt": prompt,
                "frame_index": frame_index,
                "output_prob_thresh": output_prob_thresh,
            },
        )

    def propagate_video(
        self,
        session_id: str,
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        propagation_direction: str = "forward",
        output_prob_thresh: float = 0.5,
        include_frames: bool = False,
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None
        payload: dict[str, Any] = {
            "propagation_direction": propagation_direction,
            "output_prob_thresh": output_prob_thresh,
            "include_frames": include_frames,
        }
        if start_frame_index is not None:
            payload["start_frame_index"] = start_frame_index
        if max_frame_num_to_track is not None:
            payload["max_frame_num_to_track"] = max_frame_num_to_track
        return self._post_json(f"/video/sessions/{session_id}/propagate", payload)

    def get_frame_result(self, session_id: str, frame_index: int) -> dict[str, Any] | None:
        if not self.configured:
            return None
        try:
            response = httpx.get(
                f"{self.base_url}/video/sessions/{session_id}/frames/{frame_index}",
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

    def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        if not self.configured:
            return None
        try:
            response = httpx.get(
                f"{self.base_url}/video/sessions/{session_id}/status",
                timeout=self.timeout_sec,
            )
        except httpx.HTTPError as exc:
            return {
                "source": "sam3_unavailable",
                "message": f"SAM3 request failed: {exc}",
            }
        if response.status_code >= 400:
            return {"source": "sam3_unavailable", "message": _response_message(response)}
        result = response.json()
        return result if isinstance(result, dict) else None

    def close_session(self, session_id: str) -> dict[str, Any] | None:
        if not self.configured:
            return None
        try:
            response = httpx.delete(
                f"{self.base_url}/video/sessions/{session_id}",
                timeout=self.timeout_sec,
            )
        except httpx.HTTPError as exc:
            return {
                "source": "sam3_unavailable",
                "message": f"SAM3 request failed: {exc}",
            }
        if response.status_code >= 400:
            return {"source": "sam3_unavailable", "message": _response_message(response)}
        result = response.json()
        return result if isinstance(result, dict) else None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
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
        detail = payload["detail"]
        if isinstance(detail, dict) and detail.get("message"):
            return str(detail["message"])
        return str(detail)
    return response.text
