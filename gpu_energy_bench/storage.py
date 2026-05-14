"""Persistent results store — SQLite. Keeps every run so you can compare across
power caps, sizes, dtypes, GPU configs over time.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

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
    checks TEXT
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def save_run(test_name: str | None, metrics, power_limit_w: float | None,
             gpu_name: str | None, passed: bool | None, checks: list | None) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO runs (ts, test_name, kernel, params, elapsed_s,
               gflops_per_s, total_gflops, energy_j, energy_per_gflop,
               avg_power_w, max_power_w, max_temp_c, avg_util_gpu,
               repetitions, power_limit_w, gpu_name, passed, checks)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), test_name, metrics.kernel, json.dumps(metrics.params),
                metrics.elapsed_s, metrics.gflops_per_s, metrics.total_gflops,
                metrics.energy_j, metrics.energy_per_gflop, metrics.avg_power_w,
                metrics.max_power_w, metrics.max_temp_c, metrics.avg_util_gpu,
                metrics.repetitions, power_limit_w, gpu_name,
                None if passed is None else int(passed),
                json.dumps([c.__dict__ for c in (checks or [])]),
            ),
        )
        return cur.lastrowid


def load_runs() -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY ts DESC", conn)


def clear() -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM runs")
