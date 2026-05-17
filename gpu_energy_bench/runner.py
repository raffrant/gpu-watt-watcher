"""Benchmark harness — glues kernel + telemetry + metrics together.

This is the single entry point used by the Streamlit app and (optionally) by
CLI scripts. It guarantees: warmup happened inside the kernel, telemetry is
sampled across the *entire* timed region, and metrics are returned in a
consistent shape so the test registry can evaluate thresholds uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import kernels
from .telemetry import TelemetrySampler


@dataclass
class RunMetrics:
    kernel: str
    params: dict[str, Any]
    elapsed_s: float
    gflops_per_s: float
    total_gflops: float
    energy_j: float
    energy_per_gflop: float
    avg_power_w: float
    max_power_w: float
    max_temp_c: float
    avg_util_gpu: float
    repetitions: int
    # ---- useful-work accounting -----------------------------------------
    workload_type: str = "microbenchmark"
    work_unit: str = "gflops"
    work_amount: float = 0.0
    throughput_per_s: float = 0.0
    energy_per_work_unit: float = 0.0
    latency_mean_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    checksum: float | None = None
    samples_csv: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["params"] = str(self.params)
        d["extra"] = str(self.extra)
        return d


def run(kernel_name: str, params: dict[str, Any], device: str = "cuda",
        sample_interval_s: float = 0.1, gpu_index: int = 0) -> tuple[RunMetrics, TelemetrySampler]:
    fn = kernels.get(kernel_name)

    sampler = TelemetrySampler(index=gpu_index, interval_s=sample_interval_s)
    sampler.start()
    try:
        result = fn(device=device, **params)
    finally:
        sampler.stop()

    energy_j = sampler.energy_joules()
    total_gflops = result.flops / 1e9
    energy_per_gflop = (energy_j / total_gflops) if total_gflops > 0 else float("inf")
    energy_per_work_unit = (energy_j / result.work_amount) if result.work_amount > 0 else float("inf")

    metrics = RunMetrics(
        kernel=kernel_name,
        params=params,
        elapsed_s=result.elapsed_s,
        gflops_per_s=result.gflops_per_s,
        total_gflops=total_gflops,
        energy_j=energy_j,
        energy_per_gflop=energy_per_gflop,
        avg_power_w=sampler.avg_power_w(),
        max_power_w=sampler.max_power_w(),
        max_temp_c=sampler.max_temp_c(),
        avg_util_gpu=sampler.avg_util_gpu(),
        repetitions=result.repetitions,
        workload_type=result.workload_type,
        work_unit=result.work_unit,
        work_amount=result.work_amount,
        throughput_per_s=result.throughput_per_s,
        energy_per_work_unit=energy_per_work_unit,
        latency_mean_ms=result.latency_mean_ms,
        latency_p95_ms=result.latency_p95_ms,
        checksum=result.checksum,
        extra=result.extra,
    )
    return metrics, sampler
