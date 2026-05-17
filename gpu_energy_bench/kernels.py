"""Kernel registry — tiny pluggable interface.

A kernel is any function with this signature:

    run_kernel(device: str, **params) -> KernelResult

It must do its own warmup + torch.cuda.synchronize() around the timed region.
Adding a new kernel is just: write the function, then `register("name", fn)`.

Each kernel reports the *useful work* it produced through `work_amount` and
`work_unit`. For microbenchmarks the natural unit is "gflops"; for AI training
workloads it's "samples" or "tokens"; for inference it's "tokens" or "requests".
The runner uses (energy_j / work_amount) to compute the headline
"energy per unit of useful work" metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import time


@dataclass
class KernelResult:
    elapsed_s: float
    flops: float                     # total FLOPs across all reps (0 if unknown)
    repetitions: int = 1
    checksum: Optional[float] = None
    # ---- useful-work accounting (new) -----------------------------------
    work_amount: float = 0.0         # e.g. tokens processed, samples seen
    work_unit: str = "gflops"        # "gflops" | "tokens" | "samples" | "requests"
    workload_type: str = "microbenchmark"  # "training" | "inference" | "microbenchmark"
    # ---- optional latency stats (inference) -----------------------------
    latency_mean_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Default work_amount for microbenchmarks = total GFLOPs.
        if self.work_amount == 0.0 and self.work_unit == "gflops":
            self.work_amount = self.flops / 1e9

    @property
    def gflops_per_s(self) -> float:
        return (self.flops / 1e9) / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def throughput_per_s(self) -> float:
        return self.work_amount / self.elapsed_s if self.elapsed_s > 0 else 0.0


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


def _torch_dtype(name: str):
    import torch
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _sync(dev) -> None:
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Built-in kernels
# ---------------------------------------------------------------------------

def _matmul(device: str, size: int = 4096, repetitions: int = 10,
            dtype: str = "float32", warmup: int = 2) -> KernelResult:
    """Square matmul A @ B, repeated. FLOPs per matmul = 2 * N^3."""
    import torch

    dev = torch.device(device)
    torch_dtype = _torch_dtype(dtype)
    a = torch.randn((size, size), device=dev, dtype=torch_dtype)
    b = torch.randn((size, size), device=dev, dtype=torch_dtype)

    for _ in range(warmup):
        c = a @ b
    _sync(dev)

    t0 = time.perf_counter()
    for _ in range(repetitions):
        c = a @ b
    _sync(dev)
    elapsed = time.perf_counter() - t0

    flops = 2.0 * (size ** 3) * repetitions
    checksum = float(c.float().sum().item())

    return KernelResult(
        elapsed_s=elapsed, flops=flops, repetitions=repetitions,
        checksum=checksum, work_unit="gflops",
        workload_type="microbenchmark",
        extra={"size": size, "dtype": dtype},
    )


register("matmul", _matmul)


# ---------------------------------------------------------------------------
# Training-like microbenchmark — a small Transformer encoder block
# ---------------------------------------------------------------------------

def _train_transformer(device: str, batch_size: int = 16, seq_len: int = 256,
                       d_model: int = 512, n_heads: int = 8, n_layers: int = 2,
                       steps: int = 20, dtype: str = "float32",
                       warmup: int = 2) -> KernelResult:
    """Forward+backward of a small Transformer encoder. Work unit = tokens."""
    import torch
    from torch import nn

    dev = torch.device(device)
    torch_dtype = _torch_dtype(dtype)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads,
        dim_feedforward=4 * d_model, batch_first=True,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(dev, dtype=torch_dtype)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(batch_size, seq_len, d_model, device=dev, dtype=torch_dtype)
    target = torch.randn(batch_size, seq_len, d_model, device=dev, dtype=torch_dtype)

    def _step():
        optim.zero_grad(set_to_none=True)
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optim.step()
        return float(loss.detach().to(torch.float32).item())

    for _ in range(warmup):
        _step()
    _sync(dev)

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        last_loss = _step()
    _sync(dev)
    elapsed = time.perf_counter() - t0

    tokens = float(batch_size * seq_len * steps)
    # Rough FLOP estimate: ~6 * params * tokens (fwd+bwd) for transformers.
    n_params = sum(p.numel() for p in model.parameters())
    flops = 6.0 * n_params * tokens

    return KernelResult(
        elapsed_s=elapsed, flops=flops, repetitions=steps,
        checksum=last_loss,
        work_amount=tokens, work_unit="tokens",
        workload_type="training",
        extra={"batch_size": batch_size, "seq_len": seq_len, "d_model": d_model,
               "n_heads": n_heads, "n_layers": n_layers, "dtype": dtype,
               "params": n_params, "steps_per_s": steps / elapsed if elapsed > 0 else 0.0},
    )


register("train_transformer", _train_transformer)


# ---------------------------------------------------------------------------
# Inference-like microbenchmark — token-by-token autoregressive loop
# ---------------------------------------------------------------------------

def _infer_autoregressive(device: str, batch_size: int = 4, prompt_len: int = 64,
                          gen_tokens: int = 64, d_model: int = 512,
                          n_heads: int = 8, n_layers: int = 2,
                          dtype: str = "float16", warmup: int = 1) -> KernelResult:
    """Greedy autoregressive decode over a small Transformer. Work unit = tokens."""
    import torch
    from torch import nn

    dev = torch.device(device)
    torch_dtype = _torch_dtype(dtype)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads,
        dim_feedforward=4 * d_model, batch_first=True,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(dev, dtype=torch_dtype)
    model.eval()

    @torch.inference_mode()
    def _decode():
        ctx = torch.randn(batch_size, prompt_len, d_model, device=dev, dtype=torch_dtype)
        per_token_ms: list[float] = []
        for _ in range(gen_tokens):
            t_tok = time.perf_counter()
            out = model(ctx)
            next_tok = out[:, -1:, :]
            ctx = torch.cat([ctx, next_tok], dim=1)
            _sync(dev)
            per_token_ms.append((time.perf_counter() - t_tok) * 1000.0)
        return per_token_ms, float(ctx.float().mean().item())

    for _ in range(warmup):
        _decode()
    _sync(dev)

    t0 = time.perf_counter()
    per_token_ms, checksum = _decode()
    elapsed = time.perf_counter() - t0

    tokens = float(batch_size * gen_tokens)
    n_params = sum(p.numel() for p in model.parameters())
    # Inference FLOPs ~ 2 * params * tokens.
    flops = 2.0 * n_params * tokens

    per_token_ms_sorted = sorted(per_token_ms)
    p95 = per_token_ms_sorted[int(0.95 * (len(per_token_ms_sorted) - 1))] if per_token_ms_sorted else None
    mean_ms = (sum(per_token_ms) / len(per_token_ms)) if per_token_ms else None

    return KernelResult(
        elapsed_s=elapsed, flops=flops, repetitions=gen_tokens,
        checksum=checksum,
        work_amount=tokens, work_unit="tokens",
        workload_type="inference",
        latency_mean_ms=mean_ms, latency_p95_ms=p95,
        extra={"batch_size": batch_size, "prompt_len": prompt_len,
               "gen_tokens": gen_tokens, "d_model": d_model,
               "n_heads": n_heads, "n_layers": n_layers, "dtype": dtype,
               "params": n_params},
    )


register("infer_autoregressive", _infer_autoregressive)


# ---- example template for new kernels --------------------------------------
# def _my_kernel(device, **params) -> KernelResult:
#     ...do warmup, sync, time, count flops...
#     return KernelResult(elapsed_s=..., flops=..., work_amount=..., work_unit=...)
# register("my_kernel", _my_kernel)
