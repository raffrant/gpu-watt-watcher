from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class SweepRow:
    gpu_index: int
    gpu_name: str
    power_cap_w: float
    command: str
    returncode: int
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    energy_j: float
    mean_power_w: float
    peak_power_w: float
    mean_util_pct: float
    peak_mem_mb: float
    notes: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def which_or_die(binary: str) -> None:
    if shutil.which(binary) is None:
        raise SystemExit(f"Missing dependency: {binary}")


def run_cmd(cmd, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def safe_float(v):
    s = str(v).strip()
    if s in {"N/A", "n/a", "NA", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def get_gpu_name(gpu_index: int) -> str:
    res = run_cmd([
        "nvidia-smi",
        "-i", str(gpu_index),
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    return res.stdout.strip()


def get_power_limits(gpu_index: int):
    res = run_cmd(["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "POWER"])
    mn = mx = default = None

    for line in res.stdout.splitlines():
        s = line.strip()
        lower = s.lower()
        if "min power limit" in lower and ":" in s:
            mn = safe_float(s.split(":", 1)[1].replace("W", "").strip())
        elif "max power limit" in lower and ":" in s:
            mx = safe_float(s.split(":", 1)[1].replace("W", "").strip())
        elif "default power limit" in lower and ":" in s:
            default = safe_float(s.split(":", 1)[1].replace("W", "").strip())

    return mn, mx, default


def set_power_limit(gpu_index: int, watts: float) -> None:
    run_cmd(["sudo", "nvidia-smi", "-i", str(gpu_index), "-pm", "1"])
    run_cmd(["sudo", "nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))])


def restore_power_limit(gpu_index: int, default_w: Optional[float]) -> None:
    if default_w is None:
        return
    try:
        set_power_limit(gpu_index, default_w)
    except Exception:
        pass


def sample_gpu(gpu_index: int):
    res = run_cmd([
        "nvidia-smi",
        "-i", str(gpu_index),
        "--query-gpu=power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ])
    vals = [v.strip() for v in res.stdout.strip().split(",")]
    if len(vals) != 3:
        return None
    power = safe_float(vals[0])
    util = safe_float(vals[1])
    mem_mb = safe_float(vals[2])
    if power is None:
        return None
    return power, util or 0.0, mem_mb or 0.0


def run_with_monitor(gpu_index: int, command: str, sample_period_s: float):
    samples = []
    start_ts = time.time()
    start_iso = utc_now()
    proc = subprocess.Popen(command, shell=True)
    returncode = None
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                returncode = rc
                break
            try:
                sample = sample_gpu(gpu_index)
                if sample is not None:
                    p, u, m = sample
                    samples.append((time.time(), p, u, m))
            except Exception:
                pass
            time.sleep(sample_period_s)
    finally:
        if returncode is None:
            returncode = proc.wait()
    end_iso = utc_now()
    end_ts = time.time()
    duration_s = max(end_ts - start_ts, 1e-9)

    if len(samples) >= 2:
        ts = [s[0] for s in samples]
        pw = [s[1] for s in samples]
        util = [s[2] for s in samples]
        mem = [s[3] for s in samples]
        energy_j = 0.0
        for i in range(1, len(samples)):
            dt = ts[i] - ts[i - 1]
            energy_j += 0.5 * (pw[i] + pw[i - 1]) * dt
        mean_power_w = sum(pw) / len(pw)
        peak_power_w = max(pw)
        mean_util_pct = sum(util) / len(util)
        peak_mem_mb = max(mem)
    elif len(samples) == 1:
        p, u, m = samples[0][1], samples[0][2], samples[0][3]
        mean_power_w = peak_power_w = p
        mean_util_pct = u
        peak_mem_mb = m
        energy_j = p * duration_s
    else:
        mean_power_w = peak_power_w = mean_util_pct = peak_mem_mb = energy_j = 0.0

    return {
        "returncode": returncode,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "duration_s": duration_s,
        "energy_j": energy_j,
        "mean_power_w": mean_power_w,
        "peak_power_w": peak_power_w,
        "mean_util_pct": mean_util_pct,
        "peak_mem_mb": peak_mem_mb,
        "samples": samples,
    }


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["energy_j", "duration_s"], ascending=[True, True]).reset_index(drop=True)
    keep = []
    best_time = float("inf")
    for _, r in d.iterrows():
        if r["duration_s"] < best_time:
            keep.append(True)
            best_time = r["duration_s"]
        else:
            keep.append(False)
    return d.loc[keep].copy()


def main():
    ap = argparse.ArgumentParser(description="Sweep GPU power caps and generate a Pareto report.")
    ap.add_argument("--test-cmd", required=True, help="Command to run for each power cap")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--caps", nargs="*", type=float, default=None, help="Power caps in watts")
    ap.add_argument("--outdir", default="gpu_energy_bench/output/power_sweep")
    ap.add_argument("--sample-period", type=float, default=1.0)
    ap.add_argument("--restore-default", action="store_true", default=True)
    args = ap.parse_args()

    which_or_die("nvidia-smi")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gpu_name = get_gpu_name(args.gpu_index)
    min_w, max_w, default_w = get_power_limits(args.gpu_index)

    if min_w is None or max_w is None:
        print("Power limits unavailable on this GPU; running one default benchmark.")
        caps = [default_w] if default_w is not None else [None]
    else:
        if args.caps:
            caps = [float(c) for c in args.caps if min_w <= float(c) <= max_w]
        else:
            caps = sorted({
                float(min_w),
                round(min_w + 0.33 * (max_w - min_w)),
                round(min_w + 0.66 * (max_w - min_w)),
                float(max_w),
            })

    if not caps:
        print("No valid power caps found for this GPU; using one benchmark at default state.")
        caps = [default_w] if default_w is not None else [None]

    rows = []
    telemetry_dir = outdir / "telemetry"
    telemetry_dir.mkdir(exist_ok=True)

    for cap in caps:
        if cap is not None and min_w is not None and max_w is not None:
            set_power_limit(args.gpu_index, cap)
            time.sleep(2)

        result = run_with_monitor(args.gpu_index, args.test_cmd, args.sample_period)

        row = SweepRow(
            gpu_index=args.gpu_index,
            gpu_name=gpu_name,
            power_cap_w=float(cap) if cap is not None else float("nan"),
            command=args.test_cmd,
            returncode=result["returncode"],
            start_time_utc=result["start_iso"],
            end_time_utc=result["end_iso"],
            duration_s=result["duration_s"],
            energy_j=result["energy_j"],
            mean_power_w=result["mean_power_w"],
            peak_power_w=result["peak_power_w"],
            mean_util_pct=result["mean_util_pct"],
            peak_mem_mb=result["peak_mem_mb"],
            notes=f"samples={len(result['samples'])}",
        )
        rows.append(asdict(row))

        telem = pd.DataFrame(result["samples"], columns=["ts_epoch", "power_w", "gpu_util_pct", "mem_used_mb"])
        telem.to_csv(telemetry_dir / f"gpu{args.gpu_index}_cap{str(cap).replace('.', '_')}.csv", index=False)

    if args.restore_default:
        restore_power_limit(args.gpu_index, default_w)

    df = pd.DataFrame(rows)
    cols = [
        "gpu_index", "gpu_name", "power_cap_w", "command", "returncode",
        "start_time_utc", "end_time_utc", "duration_s", "energy_j",
        "mean_power_w", "peak_power_w", "mean_util_pct", "peak_mem_mb", "notes",
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df.loc[:, cols]
    df.to_csv(outdir / "power_sweep_results.csv", index=False)
    df.to_json(outdir / "power_sweep_results.json", orient="records", indent=2)

    valid = df[df["power_cap_w"].notna()].copy()
    if valid.empty:
        valid = df.copy()
    front = pareto_front(valid[[
        "gpu_index", "gpu_name", "power_cap_w", "duration_s", "energy_j",
        "mean_power_w", "peak_power_w", "mean_util_pct", "peak_mem_mb", "returncode"
    ]].copy())
    front.to_csv(outdir / "pareto_front.csv", index=False)

    if front.empty:
        raise SystemExit("Pareto front is empty; no valid runs were recorded.")

    best = front.sort_values(["energy_j", "duration_s"]).iloc[0].to_dict()
    report = {
        "gpu_index": args.gpu_index,
        "gpu_name": gpu_name,
        "min_power_w": min_w,
        "max_power_w": max_w,
        "default_power_w": default_w,
        "tested_caps_w": caps,
        "best_pareto_cap_w": float(best["power_cap_w"]),
        "best_energy_j": float(best["energy_j"]),
        "best_duration_s": float(best["duration_s"]),
        "timestamp_utc": utc_now(),
    }
    (outdir / "pareto_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()