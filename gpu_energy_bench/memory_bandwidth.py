"""Streaming memory bandwidth benchmark with NVML energy measurement."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import pynvml
import torch


def _nvml_init():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _nvml_shutdown():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def _read(handle):
    p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    u = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return p, u.gpu, u.memory


def run_memory_bandwidth_benchmark(
    num_bytes: int,
    passes: int,
    device: str = "cuda",
    dtype=torch.float32,
) -> Dict[str, Any]:
    handle = _nvml_init()
    power_samples: List[float] = []
    gpu_util_samples: List[float] = []
    mem_util_samples: List[float] = []

    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elems = num_bytes // bytes_per_elem
    x = torch.randn(num_elems, device=device, dtype=dtype)
    x = x + 1.0
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(passes):
        x = x + 1.0
        torch.cuda.synchronize()
        p, gu, mu = _read(handle)
        power_samples.append(p)
        gpu_util_samples.append(gu)
        mem_util_samples.append(mu)
    elapsed = time.time() - t0
    _nvml_shutdown()

    bytes_moved = 2 * num_elems * bytes_per_elem * passes

    if power_samples:
        avg_power = sum(power_samples) / len(power_samples)
        avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples)
        avg_mem_util = sum(mem_util_samples) / len(mem_util_samples)
    else:
        avg_power = avg_gpu_util = avg_mem_util = 0.0

    energy_j = avg_power * elapsed
    gb_moved = bytes_moved / 1e9
    bw = gb_moved / elapsed if elapsed > 0 else 0.0
    j_per_gb = energy_j / gb_moved if gb_moved > 0 else None

    return {
        "num_bytes": num_bytes,
        "passes": passes,
        "elapsed_s": elapsed,
        "bytes_moved": bytes_moved,
        "gb_moved": gb_moved,
        "effective_bandwidth_GBps": bw,
        "avg_power_W": avg_power,
        "energy_J": energy_j,
        "energy_per_GB_J": j_per_gb,
        "avg_gpu_util_pct": avg_gpu_util,
        "avg_mem_util_pct": avg_mem_util,
    }
