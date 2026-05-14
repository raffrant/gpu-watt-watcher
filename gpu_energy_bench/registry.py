"""Test registry — pytest-like declarative tests with PASS/FAIL thresholds.

Tests live in `tests.yaml` (editable). Each test references a registered
kernel, supplies parameters, and lists thresholds. After a run, every
threshold is checked and a clear PASS/FAIL is produced.

Supported threshold keys (all optional):
  min_gflops_per_s     -> result.gflops_per_s must be >=
  max_energy_j         -> total energy (J) must be <=
  max_energy_per_gflop -> energy_j / total_gflops must be <=
  max_avg_power_w      -> sampler avg power must be <=
  max_temp_c           -> sampler max temp must be <=
  max_elapsed_s        -> wall time must be <=
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TestSpec:
    name: str
    kernel: str
    params: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    passed: bool
    actual: float
    threshold: float
    op: str  # ">=" or "<="

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.actual:.4g} {self.op} {self.threshold:.4g}"


def load_tests(path: str | Path) -> list[TestSpec]:
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    out = []
    for t in raw.get("tests", []):
        out.append(TestSpec(
            name=t["name"],
            kernel=t["kernel"],
            params=t.get("params", {}) or {},
            thresholds=t.get("thresholds", {}) or {},
        ))
    return out


def evaluate(spec: TestSpec, metrics: dict[str, float]) -> tuple[bool, list[CheckResult]]:
    """Apply spec.thresholds against the run metrics. Returns (overall_pass, checks)."""
    checks: list[CheckResult] = []

    rules = [
        ("min_gflops_per_s",     "gflops_per_s",      ">="),
        ("max_energy_j",         "energy_j",          "<="),
        ("max_energy_per_gflop", "energy_per_gflop",  "<="),
        ("max_avg_power_w",      "avg_power_w",       "<="),
        ("max_temp_c",           "max_temp_c",        "<="),
        ("max_elapsed_s",        "elapsed_s",         "<="),
    ]

    for thr_key, metric_key, op in rules:
        if thr_key not in spec.thresholds:
            continue
        threshold = float(spec.thresholds[thr_key])
        actual = float(metrics.get(metric_key, float("nan")))
        if op == ">=":
            ok = actual >= threshold
        else:
            ok = actual <= threshold
        checks.append(CheckResult(name=thr_key, passed=ok, actual=actual,
                                  threshold=threshold, op=op))

    overall = all(c.passed for c in checks) if checks else True
    return overall, checks
