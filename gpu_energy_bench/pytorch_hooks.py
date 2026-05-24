import time
import torch
import pynvml
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainMetrics:
    step_times: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    samples_processed: int = 0
    target_reached_at_s: Optional[float] = None
    target_metric_name: str = "val_loss"
    target_metric_value: float = 0.0
    _start: float = field(default_factory=time.time)

    def on_step_end(self, batch_size: int, loss: float):
        now = time.time()
        self.step_times.append(now - self._start)
        self.losses.append(loss)
        self.samples_processed += batch_size

    def on_eval(self, metric_name: str, metric_value: float):
        elapsed = time.time() - self._start
        if (
            self.target_metric_name == metric_name
            and self.target_reached_at_s is None
        ):
            # For accuracy: higher is better; for loss: lower is better
            if "acc" in metric_name and metric_value >= self.target_metric_value:
                self.target_reached_at_s = elapsed
            elif "loss" in metric_name and metric_value <= self.target_metric_value:
                self.target_reached_at_s = elapsed

    @property
    def throughput_samples_per_s(self) -> float:
        total_time = sum(self.step_times) if self.step_times else 1.0
        return self.samples_processed / max(total_time, 1e-6)

    @property
    def time_to_target_s(self) -> Optional[float]:
        return self.target_reached_at_s


def run_training_benchmark(
    train_fn,           # callable: (hooks: TrainMetrics) -> final_metric_value
    bench_run,          # BenchmarkRun instance to fill in
    target_metric_name: str = "val_loss",
    target_metric_value: float = 0.5,
    sampler=None,       # your existing NVMLSampler from bench_runner.py
):
    """Wrap a training function and populate a BenchmarkRun record."""
    import socket
    from datetime import datetime, timezone

    hooks = TrainMetrics(
        target_metric_name=target_metric_name,
        target_metric_value=target_metric_value,
    )

    bench_run.timestamp_start = datetime.now(timezone.utc).isoformat()
    bench_run.hostname = socket.gethostname()

    if sampler:
        sampler.start()

    final_metric = train_fn(hooks)

    if sampler:
        sampler.stop()
        result = sampler.result()
        bench_run.mean_power_w = result.mean_power_w
        bench_run.peak_power_w = result.peak_power_w
        bench_run.energy_joules = result.energy_j
        bench_run.mean_gpu_util = result.mean_util
        bench_run.peak_vram_gb = result.peak_mem_gb

    bench_run.timestamp_end = datetime.now(timezone.utc).isoformat()
    bench_run.throughput_samples_per_s = hooks.throughput_samples_per_s
    bench_run.final_metric_value = final_metric
    bench_run.epochs_completed = bench_run.epochs_requested
    bench_run.oom_flag = False

    return bench_run, hooks
