"""Streamlit GPU energy test bench — `pytest` for GPU energy.

Run with:    streamlit run gpu_energy_bench/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run gpu_energy_bench/streamlit_app.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from gpu_energy_bench import nvml_utils as nv
from gpu_energy_bench import kernels, runner, storage
from gpu_energy_bench.registry import load_tests, evaluate

st.set_page_config(page_title="GPU Energy Bench", layout="wide")

TESTS_PATH = Path(__file__).parent / "tests.yaml"


# ---------------------------------------------------------------------------
# Sidebar — global config
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ GPU Energy Bench")
gpu_index = st.sidebar.number_input("GPU index", 0, 16, 0, 1)
sample_hz = st.sidebar.slider("Telemetry rate (Hz)", 1, 50, 10)
sample_interval = 1.0 / sample_hz

try:
    n_devices = nv.device_count()
    st.sidebar.caption(f"NVML devices detected: {n_devices}")
except Exception as e:
    st.sidebar.error(f"NVML init failed: {e}")
    n_devices = 0


tab_info, tab_bench, tab_tests, tab_telemetry, tab_power, tab_history = st.tabs(
    ["GPU info", "Matrix benchmark", "Test registry", "Telemetry",
     "Power limits", "History"]
)


# ---------------------------------------------------------------------------
# GPU info — live snapshot (from firstbasic.py logic)
# ---------------------------------------------------------------------------
with tab_info:
    st.header("Live GPU snapshot")
    if st.button("Refresh", key="refresh_info"):
        st.rerun()
    if n_devices == 0:
        st.warning("No CUDA GPU detected by NVML.")
    else:
        s = nv.snapshot(gpu_index)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GPU", s.name)
        c2.metric("Temp (°C)", s.temperature_c if s.temperature_c is not None else "N/A")
        c3.metric("Power (W)", f"{s.power_w:.1f}" if s.power_w else "N/A")
        c4.metric("Power limit (W)", f"{s.power_limit_w:.0f}" if s.power_limit_w else "N/A")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Util GPU (%)", s.util_gpu_pct if s.util_gpu_pct is not None else "N/A")
        c2.metric("Util mem (%)", s.util_mem_pct if s.util_mem_pct is not None else "N/A")
        c3.metric("SM clock (MHz)", s.clock_sm_mhz if s.clock_sm_mhz is not None else "N/A")
        c4.metric("Mem clock (MHz)", s.clock_mem_mhz if s.clock_mem_mhz is not None else "N/A")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mem used (MB)", f"{s.mem_used_mb:.0f}" if s.mem_used_mb else "N/A")
        c2.metric("Mem total (MB)", f"{s.mem_total_mb:.0f}" if s.mem_total_mb else "N/A")
        c3.metric("Fan (%)", s.fan_pct if s.fan_pct is not None else "N/A")
        c4.metric("P-state", s.pstate or "N/A")


# ---------------------------------------------------------------------------
# Matrix benchmark — interactive single-shot run (from matrixmultipy.py)
# ---------------------------------------------------------------------------
with tab_bench:
    st.header("Matrix multiply benchmark")
    col1, col2, col3, col4 = st.columns(4)
    size = col1.number_input("Size (N×N)", 128, 16384, 4096, 128)
    reps = col2.number_input("Repetitions", 1, 1000, 10)
    dtype = col3.selectbox("dtype", ["float32", "float16", "bfloat16"])
    device = col4.selectbox("device", ["cuda", "cpu"])

    if st.button("Run benchmark", type="primary"):
        with st.spinner("Running with telemetry…"):
            metrics, sampler = runner.run(
                "matmul",
                {"size": int(size), "repetitions": int(reps), "dtype": dtype},
                device=device, sample_interval_s=sample_interval, gpu_index=gpu_index,
            )
        st.success(f"Done in {metrics.elapsed_s:.3f} s")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GFLOPs/s", f"{metrics.gflops_per_s:,.1f}")
        m2.metric("Energy (J)", f"{metrics.energy_j:,.2f}")
        m3.metric("Energy/GFLOP (J)", f"{metrics.energy_per_gflop:,.4f}")
        m4.metric("Avg power (W)", f"{metrics.avg_power_w:,.1f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max power (W)", f"{metrics.max_power_w:,.1f}")
        m2.metric("Max temp (°C)", f"{metrics.max_temp_c:.1f}")
        m3.metric("Avg util (%)", f"{metrics.avg_util_gpu:.1f}")
        m4.metric("Total GFLOPs", f"{metrics.total_gflops:,.1f}")

        # save run
        gpu_snap = nv.snapshot(gpu_index) if n_devices else None
        storage.save_run(
            test_name=None, metrics=metrics,
            power_limit_w=gpu_snap.power_limit_w if gpu_snap else None,
            gpu_name=gpu_snap.name if gpu_snap else None,
            passed=None, checks=None,
        )

        df = pd.DataFrame([s.__dict__ for s in sampler.samples])
        if not df.empty:
            st.plotly_chart(px.line(df, x="t", y="power_w", title="Power (W) during run"),
                            use_container_width=True)
            st.plotly_chart(px.line(df, x="t", y=["util_gpu", "temp_c"],
                                    title="Utilization & temperature"),
                            use_container_width=True)


# ---------------------------------------------------------------------------
# Test registry — pytest-style PASS/FAIL
# ---------------------------------------------------------------------------
with tab_tests:
    st.header("Test registry (pytest-style)")
    st.caption(f"Edit `{TESTS_PATH.name}` to add or change tests.")

    specs = load_tests(TESTS_PATH)
    if not specs:
        st.warning("No tests defined in tests.yaml.")
    else:
        names = [s.name for s in specs]
        chosen = st.multiselect("Tests to run", names, default=names)
        device = st.selectbox("Device", ["cuda", "cpu"], key="tests_device")

        if st.button("Run selected tests", type="primary"):
            results_view = []
            for spec in specs:
                if spec.name not in chosen:
                    continue
                with st.spinner(f"Running {spec.name}…"):
                    metrics, _ = runner.run(
                        spec.kernel, spec.params, device=device,
                        sample_interval_s=sample_interval, gpu_index=gpu_index,
                    )
                passed, checks = evaluate(spec, {
                    "gflops_per_s": metrics.gflops_per_s,
                    "energy_j": metrics.energy_j,
                    "energy_per_gflop": metrics.energy_per_gflop,
                    "avg_power_w": metrics.avg_power_w,
                    "max_temp_c": metrics.max_temp_c,
                    "elapsed_s": metrics.elapsed_s,
                })
                gpu_snap = nv.snapshot(gpu_index) if n_devices else None
                storage.save_run(
                    test_name=spec.name, metrics=metrics,
                    power_limit_w=gpu_snap.power_limit_w if gpu_snap else None,
                    gpu_name=gpu_snap.name if gpu_snap else None,
                    passed=passed, checks=checks,
                )
                results_view.append({
                    "test": spec.name,
                    "result": "✅ PASS" if passed else "❌ FAIL",
                    "gflops/s": round(metrics.gflops_per_s, 1),
                    "energy (J)": round(metrics.energy_j, 2),
                    "J/GFLOP": round(metrics.energy_per_gflop, 4),
                    "avg power (W)": round(metrics.avg_power_w, 1),
                    "max temp (°C)": round(metrics.max_temp_c, 1),
                    "elapsed (s)": round(metrics.elapsed_s, 3),
                    "checks": "; ".join(str(c) for c in checks) or "(no thresholds)",
                })
            if results_view:
                st.dataframe(pd.DataFrame(results_view), use_container_width=True)


# ---------------------------------------------------------------------------
# Telemetry — standalone logger (storedata.py-style)
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.header("Standalone telemetry logger")
    duration = st.number_input("Duration (s)", 1, 600, 10)
    if st.button("Sample"):
        from gpu_energy_bench.telemetry import TelemetrySampler
        import time as _t
        sampler = TelemetrySampler(index=gpu_index, interval_s=sample_interval)
        sampler.start()
        prog = st.progress(0.0)
        for i in range(int(duration * 10)):
            _t.sleep(0.1)
            prog.progress((i + 1) / (duration * 10))
        sampler.stop()
        df = pd.DataFrame([s.__dict__ for s in sampler.samples])
        st.dataframe(df.tail(20), use_container_width=True)
        if not df.empty:
            st.plotly_chart(px.line(df, x="t", y=["power_w", "temp_c", "util_gpu"]),
                            use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download telemetry CSV", csv, "telemetry.csv", "text/csv")


# ---------------------------------------------------------------------------
# Power limits — read + (optionally) set
# ---------------------------------------------------------------------------
with tab_power:
    st.header("Power limits & tuning")
    if n_devices == 0:
        st.info("No GPU.")
    else:
        info = nv.get_power_limit_info(gpu_index)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current (W)", f"{info['current_w']:.0f}" if info['current_w'] else "N/A")
        c2.metric("Min allowed (W)", f"{info['min_w']:.0f}" if info['min_w'] else "N/A")
        c3.metric("Max allowed (W)", f"{info['max_w']:.0f}" if info['max_w'] else "N/A")

        st.divider()
        st.subheader("Set power cap (uses nvidia-smi; may need sudo)")
        if info["min_w"] and info["max_w"]:
            new_pl = st.slider("New power limit (W)", int(info["min_w"]),
                               int(info["max_w"]),
                               int(info["current_w"] or info["max_w"]))
        else:
            new_pl = st.number_input("New power limit (W)", 1, 1000, 200)
        use_sudo = st.checkbox("Use sudo -n", value=False)
        if st.button("Apply power limit"):
            ok, msg = nv.set_power_limit(float(new_pl), gpu_index, use_sudo=use_sudo)
            (st.success if ok else st.error)(msg)

        st.caption(
            "💡 Energy-reduction experiment: set several power caps, run the same "
            "test under each, then compare `time` vs `energy` and `J/GFLOP` in the "
            "History tab."
        )


# ---------------------------------------------------------------------------
# History — compare runs across configs (the real payoff)
# ---------------------------------------------------------------------------
with tab_history:
    st.header("Run history & comparisons")
    df = storage.load_runs()
    if df.empty:
        st.info("No runs yet.")
    else:
        st.dataframe(df, use_container_width=True, height=300)

        st.subheader("Plots")
        # filter UI
        kernels_avail = sorted(df["kernel"].unique())
        kernel_pick = st.selectbox("Kernel", kernels_avail)
        sub = df[df["kernel"] == kernel_pick].copy()

        # extract size if present in params (best-effort)
        def _size(p):
            try:
                import json as _j
                return _j.loads(p.replace("'", '"')).get("size")
            except Exception:
                return None
        sub["size"] = sub["params"].map(_size)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.scatter(sub, x="size", y="gflops_per_s",
                                       color="power_limit_w",
                                       hover_data=["params", "test_name"],
                                       title="GFLOPs/s vs size"),
                            use_container_width=True)
            st.plotly_chart(px.scatter(sub, x="size", y="energy_j",
                                       color="power_limit_w",
                                       hover_data=["params", "test_name"],
                                       title="Energy (J) vs size"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(sub, x="elapsed_s", y="energy_j",
                                       color="power_limit_w",
                                       hover_data=["size", "params"],
                                       title="Time vs Energy (Pareto front)"),
                            use_container_width=True)
            st.plotly_chart(px.scatter(sub, x="size", y="energy_per_gflop",
                                       color="power_limit_w",
                                       hover_data=["params"],
                                       title="Energy per GFLOP vs size"),
                            use_container_width=True)

        if st.button("Clear history"):
            storage.clear()
            st.rerun()
