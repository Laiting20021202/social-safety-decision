# GPU Fix Required

This file records external GPU actions that Codex cannot complete without sudo,
driver replacement, or reboot.

## Observed Issue

The host currently reports:

- Driver: 550.163.01
- Host CUDA compatibility: 12.4
- GPU: NVIDIA GeForce RTX 4060 Ti
- VRAM: 16380 MiB

CUDA 12.8 containers fail with:

```text
unsatisfied condition: cuda>=12.8
```

CUDA 12.6 containers fail with:

```text
unsatisfied condition: cuda>=12.6
```

CUDA 12.4 containers pass `nvidia-smi`.

## Current Repository Fix

The SAM 3 CUDA image has been changed to:

- Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Python: 3.12 from deadsnakes
- PyTorch: `2.6.0+cu124`
- torchvision: `0.21.0+cu124`
- torchaudio: `2.6.0+cu124`

This avoids a host driver upgrade while keeping real Docker GPU execution
possible on the current machine.

## Optional Driver Upgrade

If CUDA 12.8 is required later, run:

```bash
sudo bash scripts/install_nvidia_runtime.sh
sudo reboot
```

Then verify:

```bash
make doctor
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Do not switch the SAM 3 image back to cu128 until the doctor confirms the host
driver supports CUDA 12.8 and Docker can start a CUDA 12.8 container.
