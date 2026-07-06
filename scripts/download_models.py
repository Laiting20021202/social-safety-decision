from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]

_CHECK_MODEL_ACCESS = ROOT / "scripts" / "check_model_access.py"
_SPEC = importlib.util.spec_from_file_location(
    "social_safety_check_model_access",
    _CHECK_MODEL_ACCESS,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load {_CHECK_MODEL_ACCESS}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MODEL_SPECS = _MODULE.MODEL_SPECS
check_model = _MODULE.check_model


ALLOW_PATTERNS: dict[str, list[str]] = {
    "sam3_image": ["sam3.pt", "*.json", "*.md"],
    "sam31_video": ["sam3.1_multiplex.pt", "*.json", "*.md"],
    "smolvlm_vqa": [
        "*.json",
        "*.txt",
        "*.model",
        "*.safetensors",
        "preprocessor_config.json",
        "tokenizer*",
        "processor*",
        "chat_template*",
        "*.py",
        "*.md",
    ],
    "robopoint_vicuna_13b": ["*.json", "*.safetensors", "*.bin", "*.md"],
    "robopoint_llama2_7b_lora": ["*.json", "*.safetensors", "*.bin", "*.md"],
    "robopoint_llama2_7b_base": ["*.json", "*.safetensors", "*.bin", "*.model", "*.md"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            files.append(path)
    return sorted(files)


def manifest_entry(
    access: dict[str, Any],
    local_dir: Path | None,
    patterns: list[str],
    downloaded_at: str,
) -> dict[str, Any]:
    files: list[str] = []
    hashes: dict[str, str] = {}
    total_bytes = 0
    if local_dir is not None and local_dir.exists():
        for path in selected_files(local_dir, patterns):
            relative = path.relative_to(local_dir).as_posix()
            files.append(relative)
            total_bytes += path.stat().st_size
            hashes[relative] = sha256_file(path)

    return {
        "model_name": access["key"],
        "repo_id": access["repo_id"],
        "revision": access.get("revision") or access.get("revision_requested"),
        "checkpoint_files": files,
        "sha256": hashes,
        "downloaded_at": downloaded_at if local_dir else None,
        "license": access.get("license"),
        "access_verified": bool(access.get("access_verified")),
        "local_dir": str(local_dir) if local_dir else None,
        "total_bytes": total_bytes,
        "status": "downloaded" if local_dir else access.get("status", "blocked"),
        "blocking_reason": access.get("blocking_reason"),
    }


def main() -> int:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    hf_home = Path(os.getenv("HF_HOME", "outputs/hf_cache")).resolve()
    os.environ["HF_HOME"] = str(hf_home)
    include_optional = os.getenv("DOWNLOAD_OPTIONAL_MODELS", "0").lower() in {"1", "true", "yes"}
    output_path = Path("outputs/model_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    checked = [check_model(api, spec, token) for spec in MODEL_SPECS]
    required_blocked = [
        item
        for item in checked
        if item.get("required_for_demo") and not item.get("access_verified")
    ]

    downloaded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_entries: list[dict[str, Any]] = []
    exit_code = 0

    for spec, access in zip(MODEL_SPECS, checked, strict=True):
        patterns = ALLOW_PATTERNS[spec.key]
        should_download = bool(access.get("access_verified")) and (
            spec.required_for_demo or include_optional
        )
        local_dir: Path | None = None
        if should_download:
            try:
                local_dir = Path(
                    snapshot_download(
                        repo_id=spec.repo_id,
                        revision=spec.revision,
                        token=token,
                        allow_patterns=patterns,
                        cache_dir=str(hf_home / "hub"),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - manifest should preserve exact failure
                access = dict(access)
                access["status"] = "blocked"
                access["access_verified"] = False
                access["blocking_reason"] = f"Download failed: {exc}"
                exit_code = 2
        elif spec.required_for_demo:
            exit_code = 2
        manifest_entries.append(manifest_entry(access, local_dir, patterns, downloaded_at))

    payload = {
        "created_at": downloaded_at,
        "hf_home": str(hf_home),
        "hf_token_present": bool(token),
        "all_required_access_verified": not required_blocked,
        "download_optional_models": include_optional,
        "models": manifest_entries,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
