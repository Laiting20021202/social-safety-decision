# GPU Setup

This project uses one RTX-class NVIDIA GPU through Docker. The SAM 3 service must
run with a CUDA container version that is no newer than the host driver CUDA
compatibility reported by `nvidia-smi`.

## Current Verified Host

Last checked: 2026-06-29

- GPU: NVIDIA GeForce RTX 4060 Ti
- VRAM: 16380 MiB
- Driver: 550.163.01
- Host CUDA compatibility: 12.4
- Docker NVIDIA runtime: installed
- NVIDIA CDI devices: available
- Secure Boot: disabled

Because this host reports CUDA compatibility 12.4, CUDA 12.8 and CUDA 12.6
containers fail before process start. The SAM 3 Dockerfile is therefore pinned to
CUDA 12.4 and PyTorch cu124 for this machine.

## Commands

Run the doctor:

```bash
make doctor
```

Expected output files:

- `outputs/doctor/host.json`
- `outputs/doctor/gpu.json`
- `outputs/doctor/docker.json`
- `outputs/doctor/cuda.json`
- `outputs/doctor/recommendations.md`

Direct CUDA container check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

SAM 3 image smoke after building:

```bash
docker run --rm --gpus all social-safety-amr-sam3-service python3.12 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)"
```

## Driver Upgrade Option

Only upgrade the host driver if you specifically need CUDA 12.6 or CUDA 12.8
containers. This requires sudo and usually a reboot:

```bash
sudo bash scripts/install_nvidia_runtime.sh
```

You can select a different package:

```bash
sudo DRIVER_PACKAGE=nvidia-driver-570-server bash scripts/install_nvidia_runtime.sh
```

After reboot, rerun `make doctor`. Do not mark SAM 3 or SAM 3.1 inference as
complete until `torch.cuda.is_available()` is true inside the SAM 3 container.
