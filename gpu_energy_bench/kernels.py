"""Kernel registry — tiny pluggable interface.

A kernel is any function with this signature:

    run_kernel(device: str, **params) -> KernelResult

It must do its own warmup + torch.cuda.synchronize() around the timed region.
Adding a new kernel is just: write the function, then `register("name", fn)`.

The default kernel `matmul` reuses the logic from matrixmultipy.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import time


@dataclass
class KernelResult:
    elapsed_s: float
    flops: float                     # total FLOPs across all reps
    repetitions: int = 1
    checksum: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def gflops_per_s(self) -> float:
        return (self.flops / 1e9) / self.elapsed_s if self.elapsed_s > 0 else 0.0


KernelFn = Callable[..., KernelResult]
_REGISTRY: dict[str, KernelFn] = {}


def register(name: str, fn: KernelFn) -> None:
    _REGISTRY[name] = fn


def get(name: str) -> KernelFn:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown kernel '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_kernels() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in kernels
# ---------------------------------------------------------------------------

def _matmul(device: str, size: int = 4096, repetitions: int = 10,
            dtype: str = "float32", warmup: int = 2) -> KernelResult:
    """Square matmul A @ B, repeated. FLOPs per matmul = 2 * N^3."""
    import torch

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    dev = torch.device(device)
    a = torch.randn((size, size), device=dev, dtype=torch_dtype)
    b = torch.randn((size, size), device=dev, dtype=torch_dtype)

    # Warmup — important so we time steady-state, not kernel compile / lazy init.
    for _ in range(warmup):
        c = a @ b
    if dev.type == "cuda":
        import torch
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repetitions):
        c = a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops = 2.0 * (size ** 3) * repetitions
    checksum = float(c.float().sum().item())  # cheap correctness anchor

    return KernelResult(
        elapsed_s=elapsed,
        flops=flops,
        repetitions=repetitions,
        checksum=checksum,
        extra={"size": size, "dtype": dtype},
    )


register("matmul", _matmul)


# ---- example template for new kernels --------------------------------------
# def _my_kernel(device, **params) -> KernelResult:
#     ...do warmup, sync, time, count flops...
#     return KernelResult(elapsed_s=..., flops=..., repetitions=..., extra={...})
# register("my_kernel", _my_kernel)
