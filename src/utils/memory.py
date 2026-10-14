"""Hardware detection and memory tracking utilities for GPU/MPS/CPU environments."""

from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import torch


@dataclass
class HardwareInfo:
    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    total_memory_mb: float
    free_memory_mb: float
    cuda_compute_capability: Optional[str] = None
    is_l4_gpu: bool = False
    supports_bf16: bool = False
    supports_4bit_bnb: bool = False


def detect_hardware() -> HardwareInfo:
    """Detect available accelerator and report memory and capabilities."""
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        total_mem = props.total_memory / (1024**2)
        
        # Free memory
        try:
            free_mem, _ = torch.cuda.mem_get_info(device_idx)
            free_mem_mb = free_mem / (1024**2)
        except Exception:
            free_mem_mb = total_mem - (torch.cuda.memory_allocated(device_idx) / (1024**2))
            
        cc = f"{props.major}.{props.minor}"
        is_l4 = "L4" in device_name.upper()
        supports_bf16 = props.major >= 8  # Ampere, Ada Lovelace, Hopper support bfloat16
        supports_bnb = True

        return HardwareInfo(
            device_type="cuda",
            device_name=device_name,
            total_memory_mb=round(total_mem, 1),
            free_memory_mb=round(free_mem_mb, 1),
            cuda_compute_capability=cc,
            is_l4_gpu=is_l4,
            supports_bf16=supports_bf16,
            supports_4bit_bnb=supports_bnb,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareInfo(
            device_type="mps",
            device_name="Apple Silicon (MPS)",
            total_memory_mb=0.0,  # Unified memory
            free_memory_mb=0.0,
            is_l4_gpu=False,
            supports_bf16=True,
            supports_4bit_bnb=False,
        )

    return HardwareInfo(
        device_type="cpu",
        device_name="Host CPU",
        total_memory_mb=0.0,
        free_memory_mb=0.0,
        is_l4_gpu=False,
        supports_bf16=False,
        supports_4bit_bnb=False,
    )


def flush_memory() -> None:
    """Aggressively collect garbage and clear accelerator memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_current_vram_mb() -> float:
    """Return currently allocated VRAM in megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def get_peak_vram_mb() -> float:
    """Return peak allocated VRAM in megabytes since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def reset_peak_vram() -> None:
    """Reset peak VRAM stats counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_execution_resources() -> Generator[Dict[str, Any], None, None]:
    """Context manager to track elapsed time and peak VRAM during an operation."""
    flush_memory()
    reset_peak_vram()
    start_vram = get_current_vram_mb()
    start_time = time.perf_counter()

    stats: Dict[str, Any] = {
        "start_vram_mb": start_vram,
        "peak_vram_mb": 0.0,
        "elapsed_sec": 0.0,
    }

    try:
        yield stats
    finally:
        stats["elapsed_sec"] = round(time.perf_counter() - start_time, 4)
        peak = get_peak_vram_mb()
        stats["peak_vram_mb"] = round(peak if peak > 0 else get_current_vram_mb(), 2)
