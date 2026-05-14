"""NVML helpers — robust wrappers that degrade gracefully when a value is N/A.

Reuses the pattern from firstbasic.py: init NVML, grab handle 0, query fields
defensively, and never crash the UI if a metric is unsupported on this GPU.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Optional

try:
    import pynvml
    _NVML_OK = True
except Exception as e:  # pragma: no cover
    pynvml = None  # type: ignore
    _NVML_OK = False
    _NVML_ERR = str(e)


_INITIALIZED = False


def _ensure_init() -> None:
    global _INITIALIZED
    if not _NVML_OK:
        raise RuntimeError(f"pynvml not available: {_NVML_ERR}")
    if not _INITIALIZED:
        pynvml.nvmlInit()
        _INITIALIZED = True


def shutdown() -> None:
    global _INITIALIZED
    if _INITIALIZED and _NVML_OK:
        try:
            pynvml.nvmlShutdown()
        finally:
            _INITIALIZED = False


def _safe(fn, default=None):
    """Call an NVML getter; return default if unsupported / N/A."""
    try:
        return fn()
    except Exception:
        return default


def get_handle(index: int = 0):
    _ensure_init()
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def device_count() -> int:
    if not _NVML_OK:
        return 0
    _ensure_init()
    return pynvml.nvmlDeviceGetCount()


@dataclass
class GpuSnapshot:
    name: str
    index: int
    temperature_c: Optional[float]
    power_w: Optional[float]
    power_limit_w: Optional[float]
    power_min_w: Optional[float]
    power_max_w: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    util_gpu_pct: Optional[float]
    util_mem_pct: Optional[float]
    clock_sm_mhz: Optional[float]
    clock_mem_mhz: Optional[float]
    fan_pct: Optional[float]
    pstate: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot(index: int = 0) -> GpuSnapshot:
    h = get_handle(index)
    name = _safe(lambda: pynvml.nvmlDeviceGetName(h), "Unknown")
    if isinstance(name, bytes):
        name = name.decode()

    temp = _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
    p_limit = _safe(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0)
    p_min, p_max = (None, None)
    constraints = _safe(lambda: pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h))
    if constraints:
        p_min = constraints[0] / 1000.0
        p_max = constraints[1] / 1000.0

    mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
    mem_used = (mem.used / 1024**2) if mem else None
    mem_total = (mem.total / 1024**2) if mem else None

    util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
    util_gpu = util.gpu if util else None
    util_mem = util.memory if util else None

    clk_sm = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    clk_mem = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))

    fan = _safe(lambda: pynvml.nvmlDeviceGetFanSpeed(h))
    pstate = _safe(lambda: f"P{pynvml.nvmlDeviceGetPerformanceState(h)}")

    return GpuSnapshot(
        name=name,
        index=index,
        temperature_c=temp,
        power_w=power,
        power_limit_w=p_limit,
        power_min_w=p_min,
        power_max_w=p_max,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
        util_gpu_pct=util_gpu,
        util_mem_pct=util_mem,
        clock_sm_mhz=clk_sm,
        clock_mem_mhz=clk_mem,
        fan_pct=fan,
        pstate=pstate,
    )


# ---- power limit control via nvidia-smi (may require sudo) -----------------

def get_power_limit_info(index: int = 0) -> dict[str, Any]:
    snap = snapshot(index)
    return {
        "current_w": snap.power_limit_w,
        "min_w": snap.power_min_w,
        "max_w": snap.power_max_w,
    }


def set_power_limit(watts: float, index: int = 0, use_sudo: bool = False) -> tuple[bool, str]:
    """Try to set power cap. Returns (ok, message)."""
    cmd = ["nvidia-smi", "-i", str(index), "-pl", str(int(watts))]
    if use_sudo:
        cmd = ["sudo", "-n"] + cmd
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return True, out.stdout.strip() or "Power limit updated."
        return False, (out.stderr or out.stdout or "Unknown error").strip()
    except FileNotFoundError:
        return False, "nvidia-smi not found on PATH."
    except Exception as e:
        return False, f"Failed: {e}"
