"""Tiny Transformer-like AI workload presets + runner (pure local, NVML)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pynvml
import torch
import torch.nn as nn


def _nvml_init():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _nvml_shutdown():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def _read_gpu_full(handle):
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "temp_C": temp,
        "power_W": power,
        "mem_used_MB": mem.used // 1024 ** 2,
        "mem_total_MB": mem.total // 1024 ** 2,
        "gpu_util_pct": util.gpu,
        "mem_util_pct": util.memory,
    }


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))


@dataclass
class AIWorkloadPreset:
    name: str
    batch_size: int
    seq_len: int
    d_model: int
    n_heads: int
    d_ff: int
    repetitions: int
    work_unit: str
    dtype: torch.dtype
    description: str


def _p(name, bs, seq, dtype, desc, d_model=512, n_heads=8, d_ff=2048, reps=20):
    return AIWorkloadPreset(name, bs, seq, d_model, n_heads, d_ff, reps,
                            "tokens", dtype, desc)


AI_PRESETS: Dict[str, AIWorkloadPreset] = {
    "trans_fp32_bs_8_seq_128":  _p("trans_fp32_bs_8_seq_128",  8,  128, torch.float32,  "FP32, batch=8, seq=128"),
    "trans_fp32_bs_32_seq_128": _p("trans_fp32_bs_32_seq_128", 32, 128, torch.float32,  "FP32, batch=32, seq=128"),
    "trans_fp32_bs_8_seq_512":  _p("trans_fp32_bs_8_seq_512",  8,  512, torch.float32,  "FP32, batch=8, seq=512"),
    "trans_fp16_bs_32_seq_128": _p("trans_fp16_bs_32_seq_128", 32, 128, torch.float16,  "FP16, batch=32, seq=128"),
    "trans_bf16_bs_32_seq_128": _p("trans_bf16_bs_32_seq_128", 32, 128, torch.bfloat16, "BF16, batch=32, seq=128"),
}


# Approx params for d_model=512, n_heads=8, d_ff=2048 ≈ 7.2 M params.
_PRESET_META = {
    "trans_fp32_bs_8_seq_128":  {"approx_params": 7_200_000, "bytes_per_param": 4},
    "trans_fp32_bs_32_seq_128": {"approx_params": 7_200_000, "bytes_per_param": 4},
    "trans_fp32_bs_8_seq_512":  {"approx_params": 7_200_000, "bytes_per_param": 4},
    "trans_fp16_bs_32_seq_128": {"approx_params": 7_200_000, "bytes_per_param": 2},
    "trans_bf16_bs_32_seq_128": {"approx_params": 7_200_000, "bytes_per_param": 2},
}


def preset_approx_params(name: str) -> int:
    return _PRESET_META.get(name, {}).get("approx_params", 0)


def preset_model_mb(name: str) -> float:
    m = _PRESET_META.get(name, {})
    return m.get("approx_params", 0) * m.get("bytes_per_param", 4) / 1e6


def run_ai_preset(preset_name: str, device: str = "cuda") -> Dict[str, Any]:
    cfg = AI_PRESETS[preset_name]
    dtype = cfg.dtype

    model = TinyTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff).to(device=device, dtype=dtype)
    criterion = nn.MSELoss()
    target = torch.zeros(cfg.batch_size, cfg.seq_len, cfg.d_model, device=device, dtype=dtype)

    # warm-up
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=device, dtype=dtype)
    criterion(model(x), target).backward()
    torch.cuda.synchronize()

    handle = _nvml_init()
    power_samples: List[float] = []
    gpu_util_samples: List[float] = []
    mem_util_samples: List[float] = []
    mem_used_samples: List[int] = []

    t0 = time.time()
    for _ in range(cfg.repetitions):
        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=device, dtype=dtype)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), target)
        loss.backward()
        torch.cuda.synchronize()

        snap = _read_gpu_full(handle)
        power_samples.append(snap["power_W"])
        gpu_util_samples.append(snap["gpu_util_pct"])
        mem_util_samples.append(snap["mem_util_pct"])
        mem_used_samples.append(snap["mem_used_MB"])
    elapsed = time.time() - t0
    _nvml_shutdown()

    if power_samples:
        avg_power = sum(power_samples) / len(power_samples)
        avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples)
        avg_mem_util = sum(mem_util_samples) / len(mem_util_samples)
        max_mem_used = max(mem_used_samples)
    else:
        avg_power = avg_gpu_util = avg_mem_util = 0.0
        max_mem_used = 0

    tokens = cfg.batch_size * cfg.seq_len * cfg.repetitions
    tps = tokens / elapsed if elapsed > 0 else 0.0
    energy_j = avg_power * elapsed
    j_per_tok = energy_j / tokens if tokens > 0 else None

    return {
        "preset_name": cfg.name,
        "description": cfg.description,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "repetitions": cfg.repetitions,
        "work_unit": cfg.work_unit,
        "dtype": str(dtype).replace("torch.", ""),
        "tokens_processed": tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": tps,
        "avg_power_W": avg_power,
        "energy_J": energy_j,
        "energy_per_token_J": j_per_tok,
        "avg_gpu_util_pct": avg_gpu_util,
        "avg_mem_util_pct": avg_mem_util,
        "max_mem_used_MB": max_mem_used,
    }
