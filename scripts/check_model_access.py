from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    required_files: tuple[str, ...]
    license_hint: str
    purpose: str
    required_for_demo: bool = True


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="sam3_image",
        repo_id="facebook/sam3",
        revision=os.getenv("SAM3_IMAGE_REVISION", "main"),
        required_files=("sam3.pt",),
        license_hint="See Hugging Face model card.",
        purpose="SAM 3 image/concept segmentation",
    ),
    ModelSpec(
        key="sam31_video",
        repo_id="facebook/sam3.1",
        revision=os.getenv("SAM3_VIDEO_REVISION", "main"),
        required_files=("sam3.1_multiplex.pt",),
        license_hint="See Hugging Face model card.",
        purpose="SAM 3.1 multiplex video tracking",
    ),
    ModelSpec(
        key="smolvlm_vqa",
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision=os.getenv("VQA_REVISION", "main"),
        required_files=("config.json",),
        license_hint="apache-2.0",
        purpose="low-frequency temporal VQA",
    ),
    ModelSpec(
        key="robopoint_vicuna_13b",
        repo_id="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        revision=os.getenv("ROBOPOINT_REVISION", "main"),
        required_files=("config.json",),
        license_hint="apache-2.0",
        purpose="RoboPoint full checkpoint option",
        required_for_demo=False,
    ),
    ModelSpec(
        key="robopoint_llama2_7b_lora",
        repo_id="wentao-yuan/robopoint-v1-llama-2-7b-lora",
        revision=os.getenv("ROBOPOINT_LORA_REVISION", "main"),
        required_files=("adapter_config.json",),
        license_hint="See Hugging Face model card.",
        purpose="RoboPoint LoRA checkpoint option for lower VRAM experiments",
        required_for_demo=False,
    ),
    ModelSpec(
        key="robopoint_llama2_7b_base",
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        revision=os.getenv("ROBOPOINT_BASE_REVISION", "main"),
        required_files=("config.json",),
        license_hint="llama2 gated license",
        purpose="base model required by the RoboPoint LLaMA-2 7B LoRA",
        required_for_demo=False,
    ),
)


def _license_from_info(info: Any, fallback: str) -> str:
    card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    if isinstance(card_data, dict):
        value = card_data.get("license")
        if value:
            return str(value)
    return fallback


def _classify_error(exc: Exception, *, token_present: bool) -> tuple[str, str]:
    if isinstance(exc, GatedRepoError):
        if token_present:
            return (
                "blocked",
                "Token is present, but gated repo file access is not granted "
                "for this account/token.",
            )
        return ("blocked", "Hugging Face access not granted. Accept the model terms and login.")
    if isinstance(exc, RepositoryNotFoundError):
        return ("blocked", "Repository not found or private for the current token.")
    if isinstance(exc, HfHubHTTPError):
        return ("blocked", f"Hugging Face HTTP error: {exc}")
    return ("blocked", f"Unexpected access error: {exc}")


def check_model(api: HfApi, spec: ModelSpec, token: str | None) -> dict[str, Any]:
    try:
        info = api.model_info(
            repo_id=spec.repo_id,
            revision=spec.revision,
            token=token,
            files_metadata=True,
        )
    except Exception as exc:  # noqa: BLE001 - preserve exact HF failure in report
        status, reason = _classify_error(exc, token_present=bool(token))
        return {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "revision_requested": spec.revision,
            "purpose": spec.purpose,
            "required_for_demo": spec.required_for_demo,
            "access_verified": False,
            "status": status,
            "blocking_reason": reason,
            "required_files": list(spec.required_files),
            "missing_files": list(spec.required_files),
            "license": spec.license_hint,
        }

    sibling_names = sorted(s.rfilename for s in info.siblings)
    missing = [name for name in spec.required_files if name not in sibling_names]
    probe_file = "config.json" if "config.json" in sibling_names else spec.required_files[0]
    probe_error: str | None = None
    if not missing:
        try:
            get_hf_file_metadata(
                hf_hub_url(spec.repo_id, probe_file, revision=spec.revision),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 - preserve exact HF failure in report
            _, probe_error = _classify_error(exc, token_present=bool(token))
    access_verified = not missing and probe_error is None
    return {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision_requested": spec.revision,
        "revision": getattr(info, "sha", None) or spec.revision,
        "purpose": spec.purpose,
        "required_for_demo": spec.required_for_demo,
        "access_verified": access_verified,
        "status": "available" if access_verified else "blocked",
        "blocking_reason": None
        if access_verified
        else probe_error or f"Required files missing from visible repo files: {', '.join(missing)}",
        "required_files": list(spec.required_files),
        "missing_files": missing,
        "access_probe_file": probe_file,
        "file_count": len(sibling_names),
        "checkpoint_files": [
            name
            for name in sibling_names
            if name.endswith((".bin", ".pt", ".pth", ".safetensors", ".json"))
        ],
        "license": _license_from_info(info, spec.license_hint),
    }


def main() -> int:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    api = HfApi()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    output_path = Path("outputs/model_access.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_identity: dict[str, Any] = {"authenticated": False, "name": None}
    if token:
        try:
            whoami = api.whoami(token=token)
            hf_identity = {
                "authenticated": True,
                "name": whoami.get("name") if isinstance(whoami, dict) else None,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            hf_identity = {"authenticated": False, "name": None, "error": str(exc)}

    results = [check_model(api, spec, token) for spec in MODEL_SPECS]
    required_blocked = [
        result
        for result in results
        if result.get("required_for_demo") and not result.get("access_verified")
    ]
    payload = {
        "checked_at": now,
        "hf_identity": hf_identity,
        "hf_token_present": bool(token),
        "all_required_access_verified": not required_blocked,
        "models": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if required_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
