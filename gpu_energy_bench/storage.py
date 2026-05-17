"""Persistent results store — SQLite. Keeps every run so you can compare across
power caps, sizes, dtypes, GPU configs over time.

Each run optionally stores its full telemetry time-series as JSON so the
History tab can re-open a saved run and re-render its plots.

The schema is extended over time via lightweight ALTER TABLE migrations so
old DBs keep working when new columns are introduced.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "results.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    test_name TEXT,
    kernel TEXT NOT NULL,
    params TEXT NOT NULL,
    elapsed_s REAL,
    gflops_per_s REAL,
    total_gflops REAL,
    energy_j REAL,
    energy_per_gflop REAL,
    avg_power_w REAL,
    max_power_w REAL,
    max_temp_c REAL,
    avg_util_gpu REAL,
    repetitions INTEGER,
    power_limit_w REAL,
    gpu_name TEXT,
    passed INTEGER,
    checks TEXT,
    telemetry_json TEXT
);
"""

# Columns added after the initial schema — name -> SQL type.
_EXTRA_COLUMNS = {
    "workload_type": "TEXT",
    "work_unit": "TEXT",
    "work_amount": "REAL",
    "throughput_per_s": "REAL",
    "energy_per_work_unit": "REAL",
    "latency_mean_ms": "REAL",
    "latency_p95_ms": "REAL",
    "driver_version": "TEXT",
    "cuda_version": "TEXT",
    "extra_json": "TEXT",
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col, sqltype in _EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {sqltype}")
    return conn


def save_run(test_name: str | None, metrics, power_limit_w: float | None,
             gpu_name: str | None, passed: bool | None, checks: list | None,
             samples_df: pd.DataFrame | None = None,
             driver_version: str | None = None,
             cuda_version: str | None = None) -> int:
    telemetry_json = None
    if samples_df is not None and not samples_df.empty:
        telemetry_json = samples_df.to_json(orient="records")

    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO runs (
                ts, test_name, kernel, params, elapsed_s,
                gflops_per_s, total_gflops, energy_j, energy_per_gflop,
                avg_power_w, max_power_w, max_temp_c, avg_util_gpu,
                repetitions, power_limit_w, gpu_name, passed, checks,
                telemetry_json,
                workload_type, work_unit, work_amount, throughput_per_s,
                energy_per_work_unit, latency_mean_ms, latency_p95_ms,
                driver_version, cuda_version, extra_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), test_name, metrics.kernel, json.dumps(metrics.params),
                metrics.elapsed_s, metrics.gflops_per_s, metrics.total_gflops,
                metrics.energy_j, metrics.energy_per_gflop, metrics.avg_power_w,
                metrics.max_power_w, metrics.max_temp_c, metrics.avg_util_gpu,
                metrics.repetitions, power_limit_w, gpu_name,
                None if passed is None else int(passed),
                json.dumps([c.__dict__ for c in (checks or [])]),
                telemetry_json,
                getattr(metrics, "workload_type", None),
                getattr(metrics, "work_unit", None),
                getattr(metrics, "work_amount", None),
                getattr(metrics, "throughput_per_s", None),
                getattr(metrics, "energy_per_work_unit", None),
                getattr(metrics, "latency_mean_ms", None),
                getattr(metrics, "latency_p95_ms", None),
                driver_version, cuda_version,
                json.dumps(getattr(metrics, "extra", {}) or {}, default=str),
            ),
        )
        return cur.lastrowid


def load_runs() -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT id, ts, test_name, kernel, params, elapsed_s, gflops_per_s, "
            "total_gflops, energy_j, energy_per_gflop, avg_power_w, max_power_w, "
            "max_temp_c, avg_util_gpu, repetitions, power_limit_w, gpu_name, "
            "passed, checks, workload_type, work_unit, work_amount, "
            "throughput_per_s, energy_per_work_unit, latency_mean_ms, "
            "latency_p95_ms, driver_version, cuda_version "
            "FROM runs ORDER BY ts DESC", conn)


def load_run(run_id: int) -> dict | None:
    """Load a single run with its telemetry + checks parsed."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (int(run_id),)
        ).fetchone()
        if not row:
            return None
        cols = [c[0] for c in conn.execute("PRAGMA table_info(runs)").fetchall()]
    record = dict(zip(cols, row))
    tj = record.get("telemetry_json")
    if tj:
        try:
            record["telemetry_df"] = pd.DataFrame(json.loads(tj))
        except Exception:
            record["telemetry_df"] = pd.DataFrame()
    else:
        record["telemetry_df"] = pd.DataFrame()
    cj = record.get("checks")
    try:
        record["checks_list"] = json.loads(cj) if cj else []
    except Exception:
        record["checks_list"] = []
    return record


def clear() -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM runs")
