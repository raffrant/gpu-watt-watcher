"""
GPU Energy Lab — local, rule-based Streamlit app
================================================
A per-machine "pytest for GPU energy". Runs matmul / memory-bandwidth /
AI-preset benchmarks, measures Joules per useful work unit (J/GFLOP,
J/GB, J/token), and emits **local, rule-based** energy advice.

NO external AI services are called. All logic is local.

Run:
    streamlit run gpu_energy_bench/gpu_energy_lab.py
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pynvml
import streamlit as st
import torch

from .ai_workloads import (
    AI_PRESETS,
    preset_approx_params,
    preset_model_mb,
    run_ai_preset,
)
from .memory_bandwidth import run_memory_bandwidth_benchmark


# =============================================================================
# NVML utilities
# =============================================================================

def nvml_init():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def nvml_shutdown():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def read_gpu_basic(handle):
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return temp, power, mem.used // 1024 ** 2, mem.total // 1024 ** 2


def read_gpu_temp_power(handle):
    t = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    return t, p


def get_gpu_env() -> Dict[str, Any]:
    handle = nvml_init()
    raw = pynvml.nvmlDeviceGetName(handle)
    name = raw.decode() if isinstance(raw, bytes) else raw
    temp, power, mem_used, mem_total = read_gpu_basic(handle)
    drv = pynvml.nvmlSystemGetDriverVersion()
    cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    nvml_shutdown()
    return {
        "gpu_name": str(name),
        "driver_version": drv.decode() if isinstance(drv, bytes) else str(drv),
        "cuda_driver_version": cuda,
        "temp_C": temp, "power_W": power,
        "mem_used_MB": mem_used, "mem_total_MB": mem_total,
    }


def show_gpu_info():
    env = get_gpu_env()
    st.subheader("Current GPU snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GPU", env["gpu_name"])
        st.metric("Driver", env["driver_version"])
    with c2:
        st.metric("Temperature (°C)", env["temp_C"])
        st.metric("Power (W)", f"{env['power_W']:.1f}")
    with c3:
        st.metric("Memory used (MB)", f"{env['mem_used_MB']} / {env['mem_total_MB']}")
        st.metric("CUDA driver", env["cuda_driver_version"])


# =============================================================================
# Telemetry logger
# =============================================================================

def log_gpu_telemetry(csv_path: Path, stop_event: threading.Event, interval_s: float = 0.2):
    handle = nvml_init()
    start = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "temp_C", "power_W", "mem_used_MB"])
        while not stop_event.is_set():
            temp, power, mem_used, _ = read_gpu_basic(handle)
            w.writerow([round(time.time() - start, 3), temp, power, mem_used])
            f.flush()
            time.sleep(interval_s)
    nvml_shutdown()


# =============================================================================
# Matmul benchmark
# =============================================================================

def run_matmul_benchmark(sizes: List[int], repetitions: int = 20) -> List[Dict]:
    handle = nvml_init()
    results = []
    for n in sizes:
        A = torch.randn(n, n, device="cuda")
        B = torch.randn(n, n, device="cuda")
        torch.matmul(A, B); torch.cuda.synchronize()

        power_samples = []
        t0 = time.time()
        for _ in range(repetitions):
            torch.matmul(A, B)
            torch.cuda.synchronize()
            power_samples.append(read_gpu_temp_power(handle)[1])
        elapsed = time.time() - t0

        temp, _ = read_gpu_temp_power(handle)
        avg_power = sum(power_samples) / len(power_samples)
        flops = 2 * n ** 3 * repetitions
        gflops_s = flops / elapsed / 1e9 if elapsed > 0 else 0.0
        energy_j = avg_power * elapsed
        epg = energy_j / (flops / 1e9) if flops > 0 else None

        results.append({
            "benchmark": "matmul",
            "n": n,
            "repetitions": repetitions,
            "elapsed_s": elapsed,
            "temp_C": temp,
            "avg_power_W": avg_power,
            "gflops_per_s": gflops_s,
            "energy_J": energy_j,
            "energy_per_GFLOP_J": epg,
        })
        time.sleep(0.3)
    nvml_shutdown()
    return results


# =============================================================================
# Rule-based energy advice (local, no external AI)
# =============================================================================

def energy_advice_matmul(row: dict) -> Tuple[str, str]:
    gflops = row.get("gflops_per_s", 0)
    epg = row.get("energy_per_GFLOP_J")
    power = row.get("avg_power_W", 0)
    n = row.get("n", 0)

    if gflops < 500 and n >= 2048:
        return ("warning",
            f"⚠️ Low compute utilisation ({gflops:.0f} GFLOPs/s for n={n}). "
            "The GPU is not being pushed hard — fuse multiple matmuls or batch "
            "more work together to amortise the fixed power overhead.")
    if epg is not None and epg > 1.0:
        return ("error",
            f"🔴 High energy per GFLOP ({epg:.2f} J/GFLOP). "
            "Try FP16/BF16 mixed precision — halving the data type typically "
            "cuts memory traffic ~2× and raises throughput, reducing J/GFLOP.")
    if power > 250:
        return ("warning",
            f"⚠️ GPU drawing {power:.0f} W. If J/GFLOP is flat as power rises, "
            "you are in a diminishing-returns zone — a 15-20% power cap may "
            "deliver the same GFLOPs/s for materially less energy.")
    return ("success",
        "✅ Matmul efficiency looks reasonable. Try larger sizes or FP16 to "
        "push J/GFLOP lower.")


def energy_advice_memory(result: dict) -> Tuple[str, str]:
    bw = result.get("effective_bandwidth_GBps", 0)
    epgb = result.get("energy_per_GB_J")
    mem_ut = result.get("avg_mem_util_pct", 0)

    if bw < 100:
        return ("warning",
            f"⚠️ Low effective bandwidth ({bw:.1f} GB/s). "
            "The kernel is not saturating the memory bus. Use larger tensors "
            "or avoid fragmented allocations to improve bus utilisation.")
    if epgb is not None and epgb > 5.0:
        return ("error",
            f"🔴 High energy per GB ({epgb:.2f} J/GB). "
            "Reducing data type from FP32 → FP16 halves bytes moved and "
            "typically cuts J/GB proportionally.")
    if mem_ut < 40:
        return ("warning",
            f"⚠️ Memory utilisation only {mem_ut:.0f}%. "
            "The memory subsystem is mostly idle — increase tensor size or "
            "concurrent streams.")
    return ("success",
        "✅ Memory bandwidth looks healthy. Profile with Nsight for "
        "cache-miss hot spots that inflate J/GB.")


def energy_advice_ai(result: dict) -> Tuple[str, str]:
    gpu_ut = result.get("avg_gpu_util_pct", result.get("avg_mem_util_pct", 50))
    mem_ut = result.get("avg_mem_util_pct", 50)
    ept = result.get("energy_per_token_J")
    tps = result.get("tokens_per_s", 0)
    dtype = result.get("dtype", "float32")
    bs = result.get("batch_size", 1)
    seq = result.get("seq_len", 128)
    pname = result.get("preset_name", "")
    model_mb = preset_model_mb(pname)
    approx_p = preset_approx_params(pname)

    lines: List[str] = []
    level = "success"

    if gpu_ut < 40:
        level = "warning"
        lines.append(
            f"⚠️ GPU compute utilisation is low ({gpu_ut:.0f}%). "
            f"With batch_size={bs} you are leaving compute on the table. "
            "Double the batch size to amortise fixed power overhead and lower "
            "energy per token.")

    if mem_ut > 80 and gpu_ut < 60:
        level = "warning"
        lines.append(
            f"⚠️ Memory-bound workload (mem_util={mem_ut:.0f}%, gpu_util={gpu_ut:.0f}%). "
            "Reduce sequence length, quantise (INT8/FP16), or use fused "
            "attention kernels (Flash-Attention) to cut DRAM traffic.")

    if "float32" in dtype and bs >= 32:
        level = "warning"
        lines.append(
            "⚠️ FP32 with large batch — switch to BF16 or FP16. "
            f"For this model (~{approx_p/1e6:.1f} M params, ~{model_mb:.0f} MB "
            f"in FP32 → ~{model_mb/2:.0f} MB in FP16) you halve memory traffic "
            "and typically improve throughput 1.5–2×, reducing J/token.")

    if ept is not None and ept > 0.005:
        level = "error"
        lines.append(
            f"🔴 Energy per token is high ({ept*1000:.3f} mJ/token). "
            "Key levers: (1) larger batch size, (2) FP16/INT8 precision, "
            "(3) smaller model via distillation.")

    if tps < 500 and seq >= 512:
        if level == "success":
            level = "warning"
        lines.append(
            f"⚠️ Low throughput ({tps:.0f} tokens/s) at seq_len={seq}. "
            "Long sequences amplify attention's O(n²) cost — consider "
            "sliding-window or linear attention.")

    if not lines:
        lines.append(
            "✅ Workload looks efficient. "
            f"Model: ~{approx_p/1e6:.1f} M params (~{model_mb:.0f} MB in {dtype}). "
            "Next: try a lower power cap (e.g. 80% of TDP) and see whether "
            "J/token improves while throughput stays acceptable.")

    return (level, "\n\n".join(lines))


def show_advice_box(level: str, text: str):
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.error(text)


# =============================================================================
# Test registry
# =============================================================================

@dataclass
class TestConfig:
    name: str
    kind: str
    params: Dict[str, Any]
    thresholds: Dict[str, float]
    description: str
    power_profile: str = "default"


TESTS: Dict[str, TestConfig] = {
    "matmul_small": TestConfig(
        name="matmul_small", kind="matmul",
        params={"sizes": [1024], "repetitions": 20},
        thresholds={"min_gflops_per_s": 1.0, "max_energy_J": 100.0},
        description="Matmul n=1024, FP32"),
    "matmul_large": TestConfig(
        name="matmul_large", kind="matmul",
        params={"sizes": [4096], "repetitions": 10},
        thresholds={"min_gflops_per_s": 1.0, "max_energy_J": 2000.0},
        description="Matmul n=4096, FP32"),
    "memory_bandwidth_1GB": TestConfig(
        name="memory_bandwidth_1GB", kind="memory_bandwidth",
        params={"num_bytes": int(1e9), "passes": 20},
        thresholds={"min_bandwidth_GBps": 50.0, "max_energy_per_GB_J": 50.0},
        description="Stream ~1 GB tensor, 20 passes"),
    "ai_fp32_bs8_seq128": TestConfig(
        name="ai_fp32_bs8_seq128", kind="ai_preset",
        params={"preset_name": "trans_fp32_bs_8_seq_128"},
        thresholds={"max_energy_per_token_J": 0.01},
        description="Tiny transformer: FP32, batch=8, seq=128"),
    "ai_fp16_bs32_seq128": TestConfig(
        name="ai_fp16_bs32_seq128", kind="ai_preset",
        params={"preset_name": "trans_fp16_bs_32_seq_128"},
        thresholds={"max_energy_per_token_J": 0.008},
        description="Tiny transformer: FP16, batch=32, seq=128"),
}


def evaluate_test(test: TestConfig) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if test.kind == "matmul":
        row = run_matmul_benchmark(**test.params)[0]
        metrics.update(row)
        metrics["pass_min_gflops_per_s"] = row["gflops_per_s"] >= test.thresholds.get("min_gflops_per_s", 0)
        metrics["pass_max_energy_J"] = row["energy_J"] <= test.thresholds.get("max_energy_J", float("inf"))
    elif test.kind == "memory_bandwidth":
        result = run_memory_bandwidth_benchmark(**test.params)
        metrics.update(result); metrics["benchmark"] = "memory_bandwidth"
        metrics["pass_min_bandwidth_GBps"] = result.get("effective_bandwidth_GBps", 0) >= test.thresholds.get("min_bandwidth_GBps", 0)
        metrics["pass_max_energy_per_GB_J"] = (result.get("energy_per_GB_J") or float("inf")) <= test.thresholds.get("max_energy_per_GB_J", float("inf"))
    elif test.kind == "ai_preset":
        result = run_ai_preset(**test.params)
        metrics.update(result); metrics["benchmark"] = "ai_preset"
        metrics["pass_max_energy_per_token_J"] = (result.get("energy_per_token_J") or float("inf")) <= test.thresholds.get("max_energy_per_token_J", float("inf"))
    else:
        raise ValueError(f"Unknown test kind: {test.kind}")
    metrics["test_name"] = test.name
    metrics["description"] = test.description
    metrics["power_profile"] = test.power_profile
    return metrics


# =============================================================================
# Unified run history
# =============================================================================

RESULTS_PATH = Path("benchmark_results.csv")


def append_run_to_history(run: Dict[str, Any], env: Dict[str, Any]) -> None:
    record = {**run, **{f"env_{k}": v for k, v in env.items()}}
    df = pd.DataFrame([record])
    df.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)


def load_history() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(RESULTS_PATH, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(RESULTS_PATH, error_bad_lines=False)


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="GPU Energy Lab", layout="wide", page_icon="⚡")

st.title("⚡ GPU Energy Lab")
st.caption(
    "Local-only GPU energy benchmark lab. Measures J/GFLOP, J/GB, J/token "
    "and gives rule-based, on-device advice. No external AI services."
)

(
    tab_info, tab_matmul, tab_memory, tab_ai,
    tab_sweep, tab_tests, tab_history, tab_telemetry,
) = st.tabs([
    "GPU info", "Matmul", "Memory bandwidth",
    "AI presets", "⚡ Power sweep", "Test suite",
    "Run history", "Telemetry viewer",
])


# ── GPU info ────────────────────────────────────────────────────────────────
with tab_info:
    st.header("GPU overview")
    if st.button("Refresh GPU snapshot", use_container_width=True):
        show_gpu_info()
    st.divider()
    st.markdown("""
**Energy optimisation levers — quick reference**

| Lever | Typical saving | How |
|---|---|---|
| Power capping (−20% TDP) | 10–20% energy, <5% perf loss | `nvidia-smi -pl <watts>` |
| FP32 → FP16 / BF16 | 30–50% J/token | `model.half()` or AMP |
| FP16 → INT8 quantisation | additional 20–30% | bitsandbytes / TensorRT |
| 2× batch size | 20–40% J/token | increase `batch_size` |
| Flash-Attention / fused kernels | 20–40% memory traffic | `scaled_dot_product_attention` |
| Knowledge distillation | proportional to size ratio | smaller student model |
""")


# ── Matmul ──────────────────────────────────────────────────────────────────
with tab_matmul:
    st.header("Matrix multiply energy benchmark")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        sizes_str = st.text_input("Matrix sizes (comma-separated)", value="256,512,1024,2048,4096")
        repetitions = st.slider("Repetitions per size", 5, 200, 20, step=5)
        run_telemetry = st.checkbox("Log telemetry to CSV while running", value=False)
        run_btn = st.button("Run matmul benchmark", type="primary")
    with col_right:
        st.markdown("**First-class metrics**")
        st.markdown("- GFLOPs/s\n- Energy (J)\n- **Energy per GFLOP (J/GFLOP)**\n- Power (W)")

    if run_btn:
        try:
            sizes = [int(x) for x in sizes_str.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid sizes; provide comma-separated integers.")
            st.stop()

        csv_path = Path("gpu_log_matmul.csv")
        stop_evt = threading.Event()
        if run_telemetry:
            threading.Thread(target=log_gpu_telemetry, args=(csv_path, stop_evt, 0.1), daemon=True).start()

        env = get_gpu_env()
        results = run_matmul_benchmark(sizes, repetitions=repetitions)

        if run_telemetry:
            stop_evt.set()
            st.success(f"Telemetry saved to {csv_path}")

        df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        for row in results:
            append_run_to_history(row, env)

        if not df.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**GFLOPs/s vs matrix size**")
                st.line_chart(df.set_index("n")["gflops_per_s"])
            with c2:
                st.markdown("**Energy (J) vs matrix size**")
                st.line_chart(df.set_index("n")["energy_J"])
            with c3:
                st.markdown("**J / GFLOP vs matrix size**")
                st.line_chart(df.set_index("n")["energy_per_GFLOP_J"])

            st.subheader("💡 Energy advice")
            worst = max(results, key=lambda r: r["n"])
            level, advice = energy_advice_matmul(worst)
            show_advice_box(level, advice)


# ── Memory bandwidth ────────────────────────────────────────────────────────
with tab_memory:
    st.header("Memory bandwidth & energy")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        num_gb = st.slider("Tensor size (GB)", 0.5, 16.0, 1.0, 0.5)
        passes = st.slider("Passes", 5, 200, 20, 5)
        run_mem_btn = st.button("Run memory bandwidth benchmark", type="primary")
    with col_right:
        st.markdown("**First-class metrics**")
        st.markdown("- Effective bandwidth (GB/s)\n- Average power (W)\n- Energy (J)\n- **Energy per GB (J/GB)**\n- Memory utilisation (%)")

    if run_mem_btn:
        env = get_gpu_env()
        result = run_memory_bandwidth_benchmark(num_bytes=int(num_gb * 1e9), passes=passes)
        result["benchmark"] = "memory_bandwidth"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bandwidth (GB/s)", f"{result['effective_bandwidth_GBps']:.1f}")
        c2.metric("Avg power (W)", f"{result['avg_power_W']:.1f}")
        c3.metric("Energy (J)", f"{result['energy_J']:.2f}")
        epgb = result.get("energy_per_GB_J")
        c4.metric("J / GB", f"{epgb:.3f}" if epgb is not None else "—")
        st.json(result)
        append_run_to_history(result, env)

        st.subheader("💡 Energy advice")
        level, advice = energy_advice_memory(result)
        show_advice_box(level, advice)


# ── AI presets ──────────────────────────────────────────────────────────────
with tab_ai:
    st.header("AI-style energy benchmarks")
    st.markdown("Run Transformer-like workloads with different batch sizes, "
                "sequence lengths, and precisions. **Energy per token** is the headline metric.")

    preset_names = list(AI_PRESETS.keys())
    preset_choice = st.selectbox(
        "Choose a preset", preset_names,
        format_func=lambda k: AI_PRESETS[k].description,
    )
    pname = preset_choice
    st.info(
        f"📐 Model size: ~{preset_approx_params(pname)/1e6:.1f} M params  |  "
        f"~{preset_model_mb(pname):.0f} MB on GPU  |  "
        f"dtype: {AI_PRESETS[pname].dtype}"
    )

    col_a, col_b = st.columns(2)
    run_single = col_a.button("Run selected preset", type="primary")
    run_all = col_b.button("Run all presets")

    results_ai: List[Dict[str, Any]] = []
    if run_single:
        env = get_gpu_env()
        res = run_ai_preset(preset_choice)
        res["benchmark"] = "ai_preset"
        append_run_to_history(res, env)
        results_ai.append(res)
    if run_all:
        env = get_gpu_env()
        progress = st.progress(0)
        for i, name in enumerate(preset_names):
            res = run_ai_preset(name)
            res["benchmark"] = "ai_preset"
            append_run_to_history(res, env)
            results_ai.append(res)
            progress.progress((i + 1) / len(preset_names))

    if results_ai:
        df_ai = pd.DataFrame(results_ai)
        st.subheader("Results")
        st.dataframe(df_ai, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**J / token by preset**")
            st.bar_chart(df_ai.set_index("preset_name")["energy_per_token_J"])
        with c2:
            st.markdown("**Tokens / s by preset**")
            st.bar_chart(df_ai.set_index("preset_name")["tokens_per_s"])
        with c3:
            if "avg_mem_util_pct" in df_ai.columns:
                st.markdown("**Memory utilisation (%) by preset**")
                st.bar_chart(df_ai.set_index("preset_name")["avg_mem_util_pct"])

        st.subheader("💡 Energy advice")
        for res in results_ai:
            with st.expander(f"Advice for: {res['description']}", expanded=len(results_ai) == 1):
                level, advice = energy_advice_ai(res)
                show_advice_box(level, advice)


# ── Power sweep ─────────────────────────────────────────────────────────────
with tab_sweep:
    st.header("⚡ Power-cap sweep experiment")
    st.markdown(
        "Run the **same AI preset at different power profiles** to find the "
        "sweet spot where a small latency cost buys a large energy reduction.\n\n"
        "> Labels here simulate the experiment structure. To apply real caps, "
        "run `sudo nvidia-smi -pl <watts>` before each profile."
    )

    sweep_preset = st.selectbox(
        "Preset for sweep", list(AI_PRESETS.keys()),
        format_func=lambda k: AI_PRESETS[k].description, key="sweep_preset",
    )
    pct_options = [100, 90, 80, 70, 60]
    selected_pcts = st.multiselect(
        "Power cap levels (% of TDP) to sweep", pct_options, default=[100, 80, 60],
    )
    run_sweep_btn = st.button("Run power sweep", type="primary")

    if run_sweep_btn and selected_pcts:
        env = get_gpu_env()
        progress = st.progress(0)
        sweep_results: List[Dict[str, Any]] = []
        for i, pct in enumerate(sorted(selected_pcts, reverse=True)):
            st.info(f"Running at {pct}% TDP label…")
            res = run_ai_preset(sweep_preset)
            res["power_profile_pct"] = pct
            res["power_profile_label"] = f"{pct}% cap"
            res["benchmark"] = "power_sweep"
            sweep_results.append(res)
            append_run_to_history(res, env)
            progress.progress((i + 1) / len(selected_pcts))

        df_sw = pd.DataFrame(sweep_results)
        st.subheader("Sweep results")
        st.dataframe(
            df_sw[["power_profile_label", "tokens_per_s", "avg_power_W",
                   "energy_J", "energy_per_token_J", "elapsed_s"]]
            .set_index("power_profile_label"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**J / token vs power profile**")
            st.bar_chart(df_sw.set_index("power_profile_label")["energy_per_token_J"])
        with c2:
            st.markdown("**Tokens / s vs power profile**")
            st.bar_chart(df_sw.set_index("power_profile_label")["tokens_per_s"])

        # Local sweet-spot heuristic
        base_tps = df_sw[df_sw["power_profile_pct"] == max(selected_pcts)]["tokens_per_s"].values[0]
        base_ept = df_sw[df_sw["power_profile_pct"] == max(selected_pcts)]["energy_per_token_J"].values[0]
        cand = df_sw[
            (df_sw["power_profile_pct"] < max(selected_pcts))
            & (df_sw["tokens_per_s"] >= 0.9 * base_tps)
        ]
        st.subheader("💡 Energy advice")
        if not cand.empty:
            sweet = cand.sort_values("energy_per_token_J").iloc[0]
            saving_pct = 100 * (base_ept - sweet["energy_per_token_J"]) / base_ept
            show_advice_box(
                "success",
                f"🎯 Sweet spot: **{sweet['power_profile_label']}** — "
                f"{saving_pct:.1f}% lower J/token with <10% throughput loss.",
            )
        else:
            show_advice_box(
                "warning",
                "No clear sweet spot within <10% throughput loss. Try lower power cap levels.",
            )


# ── Test suite ──────────────────────────────────────────────────────────────
with tab_tests:
    st.header("Energy test suite")
    test_names = list(TESTS.keys())
    selected = st.multiselect(
        "Select tests to run", test_names, default=test_names,
        format_func=lambda k: TESTS[k].description,
    )

    if st.button("Run selected tests", type="primary"):
        env = get_gpu_env()
        test_results = []
        for name in selected:
            res = evaluate_test(TESTS[name])
            append_run_to_history(res, env)
            test_results.append(res)

        if test_results:
            df_tests = pd.DataFrame(test_results)
            st.dataframe(df_tests, use_container_width=True)

            summary = []
            for r in test_results:
                flags = [v for k, v in r.items() if k.startswith("pass_") and isinstance(v, bool)]
                summary.append({"test": r["test_name"], "pass": all(flags) if flags else True})
            df_sum = pd.DataFrame(summary)
            st.markdown("**Summary**")
            st.dataframe(df_sum)

            ai_results = [r for r in test_results if r.get("benchmark") == "ai_preset"]
            if ai_results:
                st.subheader("💡 Energy advice for AI preset tests")
                for r in ai_results:
                    level, advice = energy_advice_ai(r)
                    show_advice_box(level, f"**{r['test_name']}** — {advice}")


# ── Run history ─────────────────────────────────────────────────────────────
with tab_history:
    st.header("Run history")
    df_h = load_history()
    if df_h.empty:
        st.info("No runs logged yet.")
    else:
        st.dataframe(df_h.tail(200), use_container_width=True)

        benchmarks = sorted(df_h["benchmark"].dropna().unique()) if "benchmark" in df_h.columns else []
        filt = st.multiselect("Filter by benchmark type", benchmarks, default=benchmarks)
        df_v = df_h[df_h["benchmark"].isin(filt)] if filt else df_h

        # Optional per-row filters when columns are present
        if "preset_name" in df_v.columns:
            preset_opts = sorted(df_v["preset_name"].dropna().unique())
            if preset_opts:
                preset_sel = st.multiselect("Filter by AI preset", preset_opts, default=preset_opts)
                df_v = df_v[df_v["preset_name"].isin(preset_sel) | df_v["preset_name"].isna()]
        if "test_name" in df_v.columns:
            tn_opts = sorted(df_v["test_name"].dropna().unique())
            if tn_opts:
                tn_sel = st.multiselect("Filter by test name", tn_opts, default=tn_opts)
                df_v = df_v[df_v["test_name"].isin(tn_sel) | df_v["test_name"].isna()]
        if "power_profile_label" in df_v.columns:
            pp_opts = sorted(df_v["power_profile_label"].dropna().unique())
            if pp_opts:
                pp_sel = st.multiselect("Filter by power profile", pp_opts, default=pp_opts)
                df_v = df_v[df_v["power_profile_label"].isin(pp_sel) | df_v["power_profile_label"].isna()]

        c1, c2 = st.columns(2)
        with c1:
            if {"energy_J", "elapsed_s"}.issubset(df_v.columns):
                st.markdown("**Energy vs elapsed time**")
                st.scatter_chart(df_v, x="elapsed_s", y="energy_J")
        with c2:
            if {"energy_per_token_J", "tokens_per_s"}.issubset(df_v.columns):
                df_ai_h = df_v.dropna(subset=["energy_per_token_J", "tokens_per_s"])
                if not df_ai_h.empty:
                    st.markdown("**J/token vs tokens/s (AI presets)**")
                    st.scatter_chart(df_ai_h, x="tokens_per_s", y="energy_per_token_J")


# ── Telemetry viewer ────────────────────────────────────────────────────────
with tab_telemetry:
    st.header("Telemetry CSV viewer")
    uploaded = st.file_uploader("Upload a telemetry CSV", type=["csv"])
    if uploaded:
        df_log = pd.read_csv(uploaded)
        st.write(df_log.head())
        if "time_s" in df_log.columns:
            cols = st.columns(3)
            for i, col_name in enumerate(["power_W", "temp_C", "mem_used_MB"]):
                if col_name in df_log.columns:
                    with cols[i % 3]:
                        st.markdown(f"**{col_name} over time**")
                        st.line_chart(df_log.set_index("time_s")[col_name])
