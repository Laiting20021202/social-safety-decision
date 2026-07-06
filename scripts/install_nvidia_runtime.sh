#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_nvidia_runtime.sh" >&2
  exit 1
fi

DRIVER_PACKAGE="${DRIVER_PACKAGE:-nvidia-driver-570-server}"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y "${DRIVER_PACKAGE}" nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

cat <<'EOF'
NVIDIA driver/runtime installation completed.

Reboot before rerunning:
  make doctor
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

If Secure Boot is enabled, finish MOK enrollment during reboot.
EOF
