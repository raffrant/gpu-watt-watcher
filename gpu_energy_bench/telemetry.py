"""Background telemetry sampler — same idea as storedata.py, but threaded so it
can run *during* a benchmark and produce the time-series we need to compute
energy (J = ∫ P dt) and peak/avg metrics.
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import nvml_utils as nv


@dataclass
class TelemetrySample:
    t: float           # seconds since sampler start
    power_w: float
    temp_c: float
    util_gpu: float
    util_mem: float
    mem_used_mb: float
    clock_sm_mhz: float


class TelemetrySampler:
    """Polls NVML at a fixed interval on a background thread."""

    def __init__(self, index: int = 0, interval_s: float = 0.1):
        self.index = index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[TelemetrySample] = []
        self._t0 = 0.0

    def _loop(self) -> None:
        self._t0 = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - self._t0
            try:
                s = nv.snapshot(self.index)
                self.samples.append(TelemetrySample(
                    t=t,
                    power_w=s.power_w or 0.0,
                    temp_c=s.temperature_c or 0.0,
                    util_gpu=s.util_gpu_pct or 0.0,
                    util_mem=s.util_mem_pct or 0.0,
                    mem_used_mb=s.mem_used_mb or 0.0,
                    clock_sm_mhz=s.clock_sm_mhz or 0.0,
                ))
            except Exception:
                # Never crash the sampler — skip this tick.
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[TelemetrySample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    # --- derived metrics --------------------------------------------------

    def energy_joules(self) -> float:
        """Trapezoidal integration of power(t) over the sampled window."""
        if len(self.samples) < 2:
            return 0.0
        e = 0.0
        for a, b in zip(self.samples[:-1], self.samples[1:]):
            dt = b.t - a.t
            e += 0.5 * (a.power_w + b.power_w) * dt
        return e

    def avg_power_w(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.power_w for s in self.samples) / len(self.samples)

    def max_power_w(self) -> float:
        return max((s.power_w for s in self.samples), default=0.0)

    def max_temp_c(self) -> float:
        return max((s.temp_c for s in self.samples), default=0.0)

    def avg_util_gpu(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.util_gpu for s in self.samples) / len(self.samples)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "power_w", "temp_c", "util_gpu", "util_mem",
                        "mem_used_mb", "clock_sm_mhz"])
            for s in self.samples:
                w.writerow([f"{s.t:.4f}", s.power_w, s.temp_c, s.util_gpu,
                            s.util_mem, s.mem_used_mb, s.clock_sm_mhz])
