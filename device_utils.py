"""Device selection utilities for training and inference."""
from __future__ import annotations

import torch


def _cuda_device_score(index: int) -> tuple[int, int]:
    """Return a score tuple for a CUDA device favoring more memory and SMs."""
    props = torch.cuda.get_device_properties(index)
    return props.total_memory, props.multi_processor_count


def select_device() -> torch.device:
    """Pick the best available compute device.

    Prefers the CUDA GPU with the most memory (then SM count) when multiple GPUs
    like an RTX 5090 and 4070 are present. Falls back to Apple's MPS if
    available, otherwise CPU. This keeps training on the strongest GPU while
    still running on Jetson/desktop GPUs or CPU-only Intel hosts.
    """
    if torch.cuda.is_available():
        best_index = max(range(torch.cuda.device_count()), key=_cuda_device_score)
        return torch.device(f"cuda:{best_index}")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device description for logging."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return (
            f"CUDA {device.index}: {props.name} | SMs: {props.multi_processor_count} "
            f"| Memory: {props.total_memory / (1024 ** 3):.1f} GB"
        )
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"
