from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GpuInfo:
    available: bool
    name: str
    total_mb: float
    free_mb: float
    allocated_mb: float


def gpu_info(device: str = "cuda") -> GpuInfo:
    try:
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            return GpuInfo(False, "CPU", 0.0, 0.0, 0.0)
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return GpuInfo(
            True,
            torch.cuda.get_device_name(index),
            total / 2**20,
            free / 2**20,
            torch.cuda.memory_allocated(index) / 2**20,
        )
    except Exception:
        return GpuInfo(False, "UNAVAILABLE", 0.0, 0.0, 0.0)


def release_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # PyTorch keeps one cuBLAS workspace per inference thread/handle as
            # active allocator blocks even after every tensor/model is gone.
            clear_cublas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_cublas is not None:
                clear_cublas()
            torch.cuda.empty_cache()
    except Exception:
        pass
