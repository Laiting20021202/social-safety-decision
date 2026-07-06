#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/outputs/doctor"}"

mkdir -p "$OUT_DIR"

python3 - "$ROOT_DIR" "$OUT_DIR" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SAM3_IMAGE = os.environ.get("SAM3_IMAGE", "social-safety-amr-sam3-service")


def run(command: list[str], timeout: int = 30) -> dict[str, Any]:
    found = shutil.which(command[0]) is not None
    if not found:
        return {
            "command": command,
            "found": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "found": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "found": True,
            "returncode": 124,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout}s",
        }


def write_json(name: str, payload: dict[str, Any]) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def cuda_compat_tuple(value: str | None) -> tuple[int, ...]:
    parsed = version_tuple(value)
    return parsed[:2]


def parse_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        data[key] = raw.strip().strip('"')
    return data


def parse_dockerfile_cuda() -> dict[str, str | None]:
    dockerfile = ROOT / "docker" / "Dockerfile.sam3.cuda"
    if not dockerfile.exists():
        return {"base_image": None, "cuda_version": None}
    text = dockerfile.read_text(errors="replace")
    match = re.search(r"^FROM\s+(nvidia/cuda:([0-9.]+)-[^\s]+)", text, re.MULTILINE)
    if not match:
        return {"base_image": None, "cuda_version": None}
    return {"base_image": match.group(1), "cuda_version": match.group(2)}


def parse_nvidia_smi_csv(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    row = next(csv.reader([raw.splitlines()[0]], skipinitialspace=True), None)
    if not row or len(row) < 4:
        return None
    memory_total_match = re.search(r"\d+", row[2])
    memory_used_match = re.search(r"\d+", row[3])
    return {
        "name": row[0].strip(),
        "driver_version": row[1].strip(),
        "memory_total_mib": int(memory_total_match.group(0)) if memory_total_match else None,
        "memory_used_mib": int(memory_used_match.group(0)) if memory_used_match else None,
    }


def parse_cuda_from_text(raw: str) -> str | None:
    match = re.search(r"CUDA (?:Version|version):\s*([0-9.]+)", raw)
    return match.group(1) if match else None


def docker_image_exists(image: str) -> bool:
    result = run(["docker", "image", "inspect", image], timeout=15)
    return result["returncode"] == 0


timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
dockerfile_cuda = parse_dockerfile_cuda()

host = {
    "generated_at": timestamp,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": sys.version,
    "uname": run(["uname", "-a"]),
    "os_release": parse_os_release(),
    "lspci_nvidia": run(["bash", "-lc", "lspci | grep -i nvidia"], timeout=15),
    "secure_boot": run(["bash", "-lc", "secureboot status || mokutil --sb-state || true"], timeout=15),
}

nvidia_query = run(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader",
    ],
    timeout=15,
)
nvidia_smi_raw = run(["nvidia-smi"], timeout=20)
nvidia_container = run(["nvidia-container-cli", "info"], timeout=20)
gpu_info = parse_nvidia_smi_csv(nvidia_query["stdout"])
host_cuda = parse_cuda_from_text(nvidia_smi_raw["stdout"]) or parse_cuda_from_text(
    nvidia_container["stdout"]
)

gpu = {
    "generated_at": timestamp,
    "gpu_present": host["lspci_nvidia"]["returncode"] == 0,
    "nvidia_smi_ok": nvidia_query["returncode"] == 0,
    "nvidia_smi": nvidia_smi_raw,
    "nvidia_query": nvidia_query,
    "gpu": gpu_info,
    "host_cuda_compatibility": host_cuda,
    "nvidia_kernel_module": run(["bash", "-lc", "lsmod | grep nvidia"], timeout=15),
    "nvidia_module_info": run(["bash", "-lc", "modinfo nvidia | head"], timeout=15),
    "nvidia_proc_version": run(["bash", "-lc", "cat /proc/driver/nvidia/version || true"], timeout=15),
    "nvidia_container_cli": nvidia_container,
    "packages": run(["bash", "-lc", 'dpkg -l | grep -E "nvidia-driver|nvidia-container"'], timeout=20),
}

docker_info = run(["docker", "info"], timeout=25)
docker = {
    "generated_at": timestamp,
    "docker_version": run(["docker", "version"], timeout=20),
    "docker_info": docker_info,
    "docker_runtime_lines": run(["bash", "-lc", "docker info | grep -i runtime"], timeout=20),
    "has_nvidia_runtime": " nvidia" in docker_info["stdout"] or "Runtimes: nvidia" in docker_info["stdout"],
    "has_nvidia_cdi": "nvidia.com/gpu" in docker_info["stdout"],
}

base_image = dockerfile_cuda["base_image"]
base_cuda_test = None
if base_image:
    base_cuda_test = run(["docker", "run", "--rm", "--gpus", "all", base_image, "nvidia-smi"], timeout=60)

sam3_torch_smoke = {
    "skipped": True,
    "reason": f"Docker image {SAM3_IMAGE!r} is not built.",
}
if docker_image_exists(SAM3_IMAGE):
    sam3_torch_smoke = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            SAM3_IMAGE,
            "python3.12",
            "-c",
            (
                "import json, torch; "
                "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
                "print(json.dumps({"
                "'torch': torch.__version__, "
                "'torch_cuda': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device_count': torch.cuda.device_count(), "
                "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
                "'total_memory': p.total_memory if p else None"
                "}))"
            ),
        ],
        timeout=60,
    )

container_cuda = dockerfile_cuda["cuda_version"]
cuda_version_mismatch = bool(
    cuda_compat_tuple(container_cuda)
    and cuda_compat_tuple(host_cuda)
    and cuda_compat_tuple(container_cuda) > cuda_compat_tuple(host_cuda)
)

cuda = {
    "generated_at": timestamp,
    "dockerfile_base_image": base_image,
    "dockerfile_cuda_version": container_cuda,
    "host_cuda_compatibility": host_cuda,
    "container_cuda_higher_than_host": cuda_version_mismatch,
    "base_image_gpu_smoke": base_cuda_test,
    "sam3_image": SAM3_IMAGE,
    "sam3_torch_smoke": sam3_torch_smoke,
}

recommendations: list[str] = []
if not gpu["gpu_present"]:
    recommendations.append("BLOCKED: No NVIDIA GPU was detected by lspci.")
elif not gpu["nvidia_smi_ok"]:
    recommendations.append("BLOCKED: NVIDIA GPU exists, but nvidia-smi failed. Check driver/module state.")
else:
    name = gpu_info["name"] if gpu_info else "unknown GPU"
    memory = gpu_info["memory_total_mib"] if gpu_info else "unknown"
    recommendations.append(f"GPU detected: {name}, total VRAM {memory} MiB.")

if "SecureBoot enabled" in host["secure_boot"]["stdout"]:
    recommendations.append("Secure Boot appears enabled; NVIDIA DKMS modules may require MOK enrollment.")

if not docker["has_nvidia_runtime"] and not docker["has_nvidia_cdi"]:
    recommendations.append("BLOCKED: Docker does not expose NVIDIA runtime/CDI. Install NVIDIA Container Toolkit.")

if cuda_version_mismatch:
    recommendations.append(
        f"BLOCKED: Dockerfile CUDA {container_cuda} is higher than host driver compatibility {host_cuda}. "
        "Use a CUDA base image no newer than the host compatibility or upgrade the NVIDIA driver."
    )
elif host_cuda and container_cuda:
    recommendations.append(
        f"CUDA compatibility is aligned: host supports {host_cuda}, Dockerfile requests {container_cuda}."
    )

if isinstance(sam3_torch_smoke, dict) and sam3_torch_smoke.get("returncode") == 0:
    try:
        smoke_payload = json.loads(sam3_torch_smoke["stdout"].splitlines()[-1])
    except (json.JSONDecodeError, KeyError, IndexError):
        smoke_payload = {}
    if smoke_payload.get("cuda_available") is True:
        recommendations.append(
            "SAM3 image torch CUDA smoke passed: "
            f"{smoke_payload.get('device_name')} with torch CUDA {smoke_payload.get('torch_cuda')}."
        )
    else:
        recommendations.append("BLOCKED: SAM3 image torch CUDA smoke ran but torch.cuda.is_available() was not true.")
elif isinstance(sam3_torch_smoke, dict) and not sam3_torch_smoke.get("skipped"):
    recommendations.append("BLOCKED: SAM3 image torch CUDA smoke failed. See outputs/doctor/cuda.json.")

if not recommendations:
    recommendations.append("No recommendations generated.")

write_json("host.json", host)
write_json("gpu.json", gpu)
write_json("docker.json", docker)
write_json("cuda.json", cuda)
(OUT / "recommendations.md").write_text(
    "# GPU Doctor Recommendations\n\n"
    + "\n".join(f"- {item}" for item in recommendations)
    + "\n",
)

summary = {
    "output_dir": str(OUT),
    "gpu": gpu_info,
    "host_cuda_compatibility": host_cuda,
    "dockerfile_cuda_version": container_cuda,
    "container_cuda_higher_than_host": cuda_version_mismatch,
    "recommendations": recommendations,
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
