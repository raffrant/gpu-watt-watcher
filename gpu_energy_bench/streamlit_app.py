"""Streamlit GPU energy test bench — `pytest` for GPU energy.

Run with:    streamlit run gpu_energy_bench/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import threading
import time as _t
from dataclasses import asdict
from pathlib import Path

# Allow `streamlit run gpu_energy_bench/streamlit_app.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from gpu_energy_bench import nvml_utils as nv
from gpu_energy_bench import kernels, runner, storage
from gpu_energy_bench.registry import load_tests, evaluate

st.set_page_config(page_title="GPU Energy Bench", layout="wide")

TESTS_PATH = Path(__file__).parent / "tests.yaml"

# Persist last run so the user can export it from any tab.
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # dict: {label, metrics, samples_df}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _samples_df(sampler) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in sampler.samples])


def _stash_last_run(label: str, metrics, sampler) -> None:
    st.session_state.last_run = {
        "label": label,
        "metrics": metrics,
        "samples_df": _samples_df(sampler),
    }


def _telemetry_download(samples_df: pd.DataFrame, key: str, label: str = "Download telemetry CSV"):
    if samples_df.empty:
        return
    csv = samples_df.to_csv(index=False).encode()
    st.download_button(label, csv, f"telemetry_{key}.csv", "text/csv", key=f"dl_tel_{key}")


def _export_panel(label_prefix: str = "latest"):
    """Renders an export box for the most recent run."""
    last = st.session_state.last_run
    if not last:
        st.info("Run a benchmark or test first to enable exports.")
        return

    metrics = last["metrics"]
    samples_df = last["samples_df"]
    metrics_dict = asdict(metrics) if hasattr(metrics, "__dataclass_fields__") else dict(metrics.__dict__)
    # ensure JSON-friendly
    metrics_dict["params"] = dict(metrics_dict.get("params", {}))
    metrics_dict["extra"] = dict(metrics_dict.get("extra", {}))

    st.caption(f"Latest run: **{last['label']}**")
    c1, c2, c3, c4 = st.columns(4)

    # Metrics CSV
    metrics_csv = pd.DataFrame([metrics_dict]).to_csv(index=False).encode()
    c1.download_button("Metrics CSV", metrics_csv,
                       f"metrics_{label_prefix}.csv", "text/csv",
                       key=f"dl_metrics_csv_{label_prefix}")

    # Metrics JSON
    metrics_json = json.dumps(metrics_dict, indent=2, default=str).encode()
    c2.download_button("Metrics JSON", metrics_json,
                       f"metrics_{label_prefix}.json", "application/json",
                       key=f"dl_metrics_json_{label_prefix}")

    # Telemetry CSV
    if not samples_df.empty:
        c3.download_button("Telemetry CSV",
                           samples_df.to_csv(index=False).encode(),
                           f"telemetry_{label_prefix}.csv", "text/csv",
                           key=f"dl_tel_csv_{label_prefix}")

        # Plotly HTML (power + util + temp)
        fig = px.line(samples_df, x="t", y=["power_w", "util_gpu", "temp_c"],
                      title=f"Run telemetry — {last['label']}")
        html_buf = io.StringIO()
        pio.write_html(fig, file=html_buf, include_plotlyjs="cdn", full_html=True)
        c4.download_button("Plotly HTML", html_buf.getvalue().encode(),
                           f"telemetry_{label_prefix}.html", "text/html",
                           key=f"dl_tel_html_{label_prefix}")
    else:
        c3.caption("No telemetry samples captured.")


def _render_check_panel(checks):
    """Big visible PASS/FAIL panel for each threshold."""
    if not checks:
        st.info("This test defines no thresholds — nothing to evaluate.")
        return
    overall = all(c.passed for c in checks)
    if overall:
        st.success("✅ ALL THRESHOLDS PASSED")
    else:
        st.error("❌ ONE OR MORE THRESHOLDS FAILED")

    cols = st.columns(min(len(checks), 3) or 1)
    for i, c in enumerate(checks):
        with cols[i % len(cols)]:
            if c.passed:
                st.success(f"✅ **{c.name}**\n\n"
                           f"actual `{c.actual:.4g}` {c.op} threshold `{c.threshold:.4g}`")
            else:
                st.error(f"❌ **{c.name}**\n\n"
                         f"actual `{c.actual:.4g}` not {c.op} threshold `{c.threshold:.4g}`")


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


tab_info, tab_bench, tab_tests, tab_telemetry, tab_power, tab_history, tab_export = st.tabs(
    ["GPU info", "Matrix benchmark", "Test registry", "Telemetry",
     "Power limits", "History", "Export"]
)


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
with tab_info:
    st.header("Live GPU snapshot")
    cA, cB = st.columns([1, 3])
    live = cA.toggle("Live refresh", value=True, key="info_live")
    refresh_hz = cB.slider("Refresh rate (Hz)", 1, 10, 2, key="info_hz")

    if n_devices == 0:
        st.warning("No CUDA GPU detected by NVML.")
    else:
        @st.fragment(run_every=(1.0 / refresh_hz) if live else None)
        def _gpu_panel():
            s = nv.snapshot(gpu_index)
            st.caption(f"Updated at {_t.strftime('%H:%M:%S')}")
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

        _gpu_panel()


# ---------------------------------------------------------------------------
# Matrix benchmark
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
        _stash_last_run(f"matmul size={size} dtype={dtype} reps={reps}", metrics, sampler)

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

        gpu_snap = nv.snapshot(gpu_index) if n_devices else None
        storage.save_run(
            test_name=None, metrics=metrics,
            power_limit_w=gpu_snap.power_limit_w if gpu_snap else None,
            gpu_name=gpu_snap.name if gpu_snap else None,
            passed=None, checks=None,
        )

        df = _samples_df(sampler)
        if not df.empty:
            st.plotly_chart(px.line(df, x="t", y="power_w", title="Power (W) during run"),
                            use_container_width=True)
            st.plotly_chart(px.line(df, x="t", y=["util_gpu", "temp_c"],
                                    title="Utilization & temperature"),
                            use_container_width=True)
            _telemetry_download(df, key="bench")

        st.divider()
        st.subheader("Export this run")
        _export_panel("bench")


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
with tab_tests:
    st.header("Test registry (pytest-style)")
    st.caption(f"Defined in `{TESTS_PATH.name}`. You can override params here before running.")

    specs = load_tests(TESTS_PATH)
    if not specs:
        st.warning("No tests defined in tests.yaml.")
    else:
        names = [s.name for s in specs]
        chosen_name = st.selectbox("Test to edit & run", names)
        spec = next(s for s in specs if s.name == chosen_name)

        st.markdown("**Override parameters for this run** (does not modify tests.yaml)")
        p = dict(spec.params)
        c1, c2, c3 = st.columns(3)
        size_v = c1.number_input("size", 128, 16384,
                                 int(p.get("size", 4096)), 128, key="t_size")
        reps_v = c2.number_input("repetitions", 1, 1000,
                                 int(p.get("repetitions", 10)), key="t_reps")
        dtype_default = str(p.get("dtype", "float32"))
        dtype_options = ["float32", "float16", "bfloat16"]
        dtype_idx = dtype_options.index(dtype_default) if dtype_default in dtype_options else 0
        dtype_v = c3.selectbox("dtype", dtype_options, index=dtype_idx, key="t_dtype")

        device = st.selectbox("Device", ["cuda", "cpu"], key="tests_device")

        with st.expander("Thresholds for this test"):
            st.json(spec.thresholds or {})

        if st.button("Run selected test", type="primary"):
            params = {**p, "size": int(size_v),
                      "repetitions": int(reps_v), "dtype": dtype_v}
            with st.spinner(f"Running {spec.name}…"):
                metrics, sampler = runner.run(
                    spec.kernel, params, device=device,
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
            _stash_last_run(f"test:{spec.name}", metrics, sampler)

            gpu_snap = nv.snapshot(gpu_index) if n_devices else None
            storage.save_run(
                test_name=spec.name, metrics=metrics,
                power_limit_w=gpu_snap.power_limit_w if gpu_snap else None,
                gpu_name=gpu_snap.name if gpu_snap else None,
                passed=passed, checks=checks,
            )

            st.subheader(f"Result: {spec.name}")
            _render_check_panel(checks)

            st.markdown("**Run metrics**")
            mdf = pd.DataFrame([{
                "gflops/s": round(metrics.gflops_per_s, 1),
                "energy (J)": round(metrics.energy_j, 2),
                "J/GFLOP": round(metrics.energy_per_gflop, 4),
                "avg power (W)": round(metrics.avg_power_w, 1),
                "max temp (°C)": round(metrics.max_temp_c, 1),
                "elapsed (s)": round(metrics.elapsed_s, 3),
            }])
            st.dataframe(mdf, use_container_width=True)

            df = _samples_df(sampler)
            if not df.empty:
                st.plotly_chart(px.line(df, x="t", y="power_w",
                                        title="Power (W) during run"),
                                use_container_width=True)
                _telemetry_download(df, key=f"test_{spec.name}")

            st.divider()
            st.subheader("Export this run")
            _export_panel(f"test_{spec.name}")


# ---------------------------------------------------------------------------
# Telemetry — standalone logger
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.header("Telemetry — live & standalone")
    from gpu_energy_bench.telemetry import TelemetrySampler

    mode = st.radio("Mode", ["Standalone idle logger", "Live during benchmark"],
                    horizontal=True, key="tel_mode")

    if mode == "Standalone idle logger":
        duration = st.number_input("Duration (s)", 1, 600, 10)
        if st.button("Sample"):
            sampler = TelemetrySampler(index=gpu_index, interval_s=sample_interval)
            sampler.start()
            chart_slot = st.empty()
            prog = st.progress(0.0)
            n_ticks = max(1, int(duration * 5))
            for i in range(n_ticks):
                _t.sleep(0.2)
                prog.progress((i + 1) / n_ticks)
                df_live = _samples_df(sampler)
                if not df_live.empty:
                    chart_slot.plotly_chart(
                        px.line(df_live, x="t", y=["power_w", "temp_c", "util_gpu"],
                                title="Live telemetry"),
                        use_container_width=True, key=f"idle_live_{i}")
            sampler.stop()
            df = _samples_df(sampler)
            st.dataframe(df.tail(20), use_container_width=True)
            if not df.empty:
                _telemetry_download(df, key="standalone")

    else:
        st.caption("Runs a kernel in a background thread and streams telemetry "
                   "in real time. The sampler starts/stops exactly with the kernel.")
        specs = load_tests(TESTS_PATH)
        live_kernel = st.selectbox("Kernel",
                                   sorted(set([s.kernel for s in specs] + ["matmul"])),
                                   key="live_kernel")
        c1, c2, c3 = st.columns(3)
        l_size = c1.number_input("size", 128, 16384, 4096, 128, key="live_size")
        l_reps = c2.number_input("repetitions", 1, 1000, 20, key="live_reps")
        l_dtype = c3.selectbox("dtype", ["float32", "float16", "bfloat16"], key="live_dtype")
        l_device = st.selectbox("device", ["cuda", "cpu"], key="live_device")

        if st.button("▶ Run with live telemetry", type="primary"):
            params = {"size": int(l_size), "repetitions": int(l_reps), "dtype": l_dtype}
            sampler = TelemetrySampler(index=gpu_index, interval_s=sample_interval)
            result_box: dict = {}

            def _runner_thread():
                try:
                    fn = kernels.get(live_kernel)
                    sampler.start()
                    res = fn(device=l_device, **params)
                    result_box["result"] = res
                except Exception as ex:
                    result_box["error"] = ex
                finally:
                    sampler.stop()

            th = threading.Thread(target=_runner_thread, daemon=True)
            th.start()

            status = st.info("⏳ Running — streaming telemetry…")
            power_slot = st.empty()
            temp_slot = st.empty()
            util_slot = st.empty()
            clock_slot = st.empty()
            metric_slot = st.empty()

            tick = 0
            while th.is_alive():
                _t.sleep(0.25)
                tick += 1
                df_live = _samples_df(sampler)
                if df_live.empty:
                    continue
                with metric_slot.container():
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Samples", len(df_live))
                    mc2.metric("Now power (W)", f"{df_live['power_w'].iloc[-1]:.1f}")
                    mc3.metric("Peak power (W)", f"{df_live['power_w'].max():.1f}")
                    mc4.metric("Now temp (°C)", f"{df_live['temp_c'].iloc[-1]:.1f}")
                power_slot.plotly_chart(
                    px.line(df_live, x="t", y="power_w", title="Power (W) — live"),
                    use_container_width=True, key=f"lp_p_{tick}")
                temp_slot.plotly_chart(
                    px.line(df_live, x="t", y="temp_c", title="Temperature (°C) — live"),
                    use_container_width=True, key=f"lp_t_{tick}")
                util_slot.plotly_chart(
                    px.line(df_live, x="t", y=["util_gpu", "util_mem"],
                            title="Utilization (%) — live"),
                    use_container_width=True, key=f"lp_u_{tick}")
                clock_slot.plotly_chart(
                    px.line(df_live, x="t", y="clock_sm_mhz",
                            title="SM clock (MHz) — live"),
                    use_container_width=True, key=f"lp_c_{tick}")

            th.join()
            if "error" in result_box:
                status.error(f"Kernel failed: {result_box['error']}")
            else:
                res = result_box["result"]
                energy_j = sampler.energy_joules()
                status.success(
                    f"✅ Done in {res.elapsed_s:.3f} s — "
                    f"{res.gflops_per_s:,.1f} GFLOPs/s, energy {energy_j:,.2f} J"
                )
                # Build a RunMetrics-compatible record so Export works.
                from gpu_energy_bench.runner import RunMetrics
                total_gflops = res.flops / 1e9
                metrics = RunMetrics(
                    kernel=live_kernel, params=params,
                    elapsed_s=res.elapsed_s, gflops_per_s=res.gflops_per_s,
                    total_gflops=total_gflops, energy_j=energy_j,
                    energy_per_gflop=(energy_j / total_gflops) if total_gflops > 0 else float("inf"),
                    avg_power_w=sampler.avg_power_w(), max_power_w=sampler.max_power_w(),
                    max_temp_c=sampler.max_temp_c(), avg_util_gpu=sampler.avg_util_gpu(),
                    repetitions=res.repetitions, checksum=res.checksum, extra=res.extra,
                )
                _stash_last_run(f"live:{live_kernel} size={l_size} dtype={l_dtype}",
                                metrics, sampler)
                df_final = _samples_df(sampler)
                _telemetry_download(df_final, key="live_run")


# ---------------------------------------------------------------------------
# Power limits + sweep
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
        use_sudo_single = st.checkbox("Use sudo -n", value=False, key="sudo_single")
        if st.button("Apply power limit"):
            ok, msg = nv.set_power_limit(float(new_pl), gpu_index, use_sudo=use_sudo_single)
            (st.success if ok else st.error)(msg)

        # ------------------------------------------------------------
        # Power-cap sweep
        # ------------------------------------------------------------
        st.divider()
        st.subheader("⚡ Power-cap sweep")
        st.caption(
            "Run the same test under multiple power caps and plot time vs energy. "
            "Requires permission to run `nvidia-smi -pl` (toggle sudo if needed)."
        )

        specs = load_tests(TESTS_PATH)
        if not specs:
            st.warning("Define at least one test in tests.yaml to use sweep.")
        else:
            spec_names = [s.name for s in specs]
            sweep_test = st.selectbox("Test", spec_names, key="sweep_test")
            spec = next(s for s in specs if s.name == sweep_test)

            min_w = int(info["min_w"] or 100)
            max_w = int(info["max_w"] or 300)
            cur_w = int(info["current_w"] or max_w)

            colA, colB, colC = st.columns(3)
            sweep_low = colA.number_input("Min cap (W)", min_w, max_w, min_w, key="sw_lo")
            sweep_high = colB.number_input("Max cap (W)", min_w, max_w, max_w, key="sw_hi")
            sweep_steps = colC.number_input("Steps", 2, 20, 4, key="sw_steps")

            sweep_device = st.selectbox("Device", ["cuda", "cpu"], key="sweep_dev")
            use_sudo_sweep = st.checkbox("Use sudo -n for nvidia-smi",
                                         value=False, key="sudo_sweep")
            restore_after = st.checkbox("Restore original cap when done",
                                        value=True, key="sw_restore")

            if st.button("Run power-cap sweep", type="primary"):
                if sweep_high < sweep_low:
                    st.error("Max cap must be >= min cap.")
                else:
                    caps = [int(round(sweep_low + i * (sweep_high - sweep_low) /
                                      (sweep_steps - 1)))
                            for i in range(int(sweep_steps))]
                    rows = []
                    last_sampler = None
                    last_metrics = None
                    prog = st.progress(0.0)
                    status = st.empty()
                    gpu_snap = nv.snapshot(gpu_index)

                    for i, cap in enumerate(caps):
                        status.info(f"Setting cap to {cap} W…")
                        ok, msg = nv.set_power_limit(float(cap), gpu_index,
                                                     use_sudo=use_sudo_sweep)
                        if not ok:
                            st.warning(f"Could not set cap {cap} W: {msg}. "
                                       "Recording run anyway under current cap.")
                        _t.sleep(1.0)  # let driver settle
                        status.info(f"[{i+1}/{len(caps)}] Running at cap≈{cap} W…")
                        metrics, sampler = runner.run(
                            spec.kernel, spec.params, device=sweep_device,
                            sample_interval_s=sample_interval, gpu_index=gpu_index,
                        )
                        last_sampler, last_metrics = sampler, metrics
                        rows.append({
                            "cap_w": cap,
                            "elapsed_s": metrics.elapsed_s,
                            "energy_j": metrics.energy_j,
                            "gflops_per_s": metrics.gflops_per_s,
                            "energy_per_gflop": metrics.energy_per_gflop,
                            "avg_power_w": metrics.avg_power_w,
                            "max_power_w": metrics.max_power_w,
                            "max_temp_c": metrics.max_temp_c,
                        })
                        storage.save_run(
                            test_name=f"sweep:{spec.name}", metrics=metrics,
                            power_limit_w=cap, gpu_name=gpu_snap.name,
                            passed=None, checks=None,
                        )
                        prog.progress((i + 1) / len(caps))

                    if restore_after and gpu_snap.power_limit_w:
                        nv.set_power_limit(float(gpu_snap.power_limit_w),
                                           gpu_index, use_sudo=use_sudo_sweep)
                        status.success(f"Sweep done. Restored cap to "
                                       f"{int(gpu_snap.power_limit_w)} W.")
                    else:
                        status.success("Sweep done.")

                    sdf = pd.DataFrame(rows)
                    st.dataframe(sdf, use_container_width=True)

                    st.plotly_chart(px.scatter(sdf, x="elapsed_s", y="energy_j",
                                               text="cap_w", size="cap_w",
                                               title="Time vs Energy across power caps"
                                                     " (Pareto)").update_traces(
                                                         textposition="top center"),
                                    use_container_width=True)
                    st.plotly_chart(px.line(sdf.sort_values("cap_w"), x="cap_w",
                                            y=["elapsed_s", "energy_j"],
                                            title="Cap vs time & energy"),
                                    use_container_width=True)
                    st.plotly_chart(px.line(sdf.sort_values("cap_w"), x="cap_w",
                                            y="energy_per_gflop",
                                            title="Energy per GFLOP vs cap"),
                                    use_container_width=True)

                    st.download_button("Download sweep CSV",
                                       sdf.to_csv(index=False).encode(),
                                       "power_sweep.csv", "text/csv")

                    if last_sampler is not None and last_metrics is not None:
                        _stash_last_run(f"sweep:{spec.name} "
                                        f"@cap={rows[-1]['cap_w']}W",
                                        last_metrics, last_sampler)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
with tab_history:
    st.header("Run history & comparisons")
    df = storage.load_runs()
    if df.empty:
        st.info("No runs yet.")
    else:
        # ---- filters ----
        st.subheader("Filters")
        f1, f2, f3 = st.columns(3)
        test_names = sorted([t for t in df["test_name"].dropna().unique().tolist()])
        test_pick = f1.multiselect("Test name", test_names, default=test_names)
        kernels_avail = sorted(df["kernel"].dropna().unique().tolist())
        kernel_pick = f2.multiselect("Kernel", kernels_avail, default=kernels_avail)
        params_q = f3.text_input("Params contains (substring)", "")

        cap_series = df["power_limit_w"].dropna()
        if not cap_series.empty:
            cap_lo, cap_hi = float(cap_series.min()), float(cap_series.max())
            if cap_lo == cap_hi:
                cap_hi = cap_lo + 1.0
            cap_range = st.slider("Power cap range (W)", cap_lo, cap_hi,
                                  (cap_lo, cap_hi))
        else:
            cap_range = None

        regr_pct = st.slider(
            "Regression threshold (% worse than previous run of same "
            "test+params+cap)", 1, 100, 10
        )

        sub = df.copy()
        if test_pick:
            sub = sub[sub["test_name"].isin(test_pick) | sub["test_name"].isna()
                      if len(test_pick) == len(test_names)
                      else sub["test_name"].isin(test_pick)]
        if kernel_pick:
            sub = sub[sub["kernel"].isin(kernel_pick)]
        if params_q:
            sub = sub[sub["params"].str.contains(params_q, case=False, na=False)]
        if cap_range is not None:
            sub = sub[(sub["power_limit_w"].fillna(-1) >= cap_range[0]) &
                      (sub["power_limit_w"].fillna(-1) <= cap_range[1])]

        # ---- regression detection ----
        # Sort chronologically per (test_name, params, power_limit_w),
        # mark a row as a regression if energy_j or elapsed_s exceeds
        # the previous run's value by >regr_pct%.
        sub = sub.sort_values("ts")
        group_keys = ["test_name", "params", "power_limit_w"]
        sub["prev_energy_j"] = sub.groupby(group_keys)["energy_j"].shift(1)
        sub["prev_elapsed_s"] = sub.groupby(group_keys)["elapsed_s"].shift(1)
        thr = 1 + regr_pct / 100.0
        sub["energy_regression"] = (
            sub["prev_energy_j"].notna() & (sub["energy_j"] > sub["prev_energy_j"] * thr)
        )
        sub["time_regression"] = (
            sub["prev_elapsed_s"].notna() & (sub["elapsed_s"] > sub["prev_elapsed_s"] * thr)
        )
        sub["regression"] = sub["energy_regression"] | sub["time_regression"]
        sub = sub.sort_values("ts", ascending=False)

        n_regr = int(sub["regression"].sum())
        if n_regr:
            st.error(f"⚠️ {n_regr} regression(s) detected (>{regr_pct}% worse "
                     "than previous comparable run).")
        else:
            st.success("✅ No regressions in the filtered runs.")

        st.subheader(f"Filtered runs ({len(sub)})")

        def _row_style(row):
            if row.get("regression"):
                return ["background-color: rgba(220, 38, 38, 0.18)"] * len(row)
            if row.get("passed") == 0:
                return ["background-color: rgba(234, 179, 8, 0.18)"] * len(row)
            return [""] * len(row)

        display_cols = [
            "ts", "test_name", "kernel", "params", "power_limit_w",
            "elapsed_s", "energy_j", "energy_per_gflop", "gflops_per_s",
            "avg_power_w", "max_temp_c", "passed",
            "prev_energy_j", "prev_elapsed_s", "regression",
        ]
        display_cols = [c for c in display_cols if c in sub.columns]
        styled = sub[display_cols].style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, height=320)
        st.download_button("Download filtered history CSV",
                           sub.to_csv(index=False).encode(),
                           "history_filtered.csv", "text/csv")

        # ---- comparison plots ----
        st.subheader("Energy vs time across filtered runs")
        if not sub.empty:
            sub["size"] = sub["params"].map(lambda p: (
                json.loads(p.replace("'", '"')).get("size")
                if isinstance(p, str) else None
            ) if p else None)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.scatter(sub, x="elapsed_s", y="energy_j",
                               color="power_limit_w", symbol="regression",
                               hover_data=["test_name", "params", "ts"],
                               title="Time vs Energy (Pareto — × = regression)"),
                    use_container_width=True)
                st.plotly_chart(
                    px.scatter(sub, x="size", y="gflops_per_s",
                               color="power_limit_w",
                               hover_data=["test_name", "params"],
                               title="GFLOPs/s vs size"),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    px.line(sub.sort_values("ts"), x="ts", y="energy_j",
                            color="test_name",
                            title="Energy (J) over time per test"),
                    use_container_width=True)
                st.plotly_chart(
                    px.line(sub.sort_values("ts"), x="ts", y="elapsed_s",
                            color="test_name",
                            title="Elapsed (s) over time per test"),
                    use_container_width=True)

        if st.button("Clear history"):
            storage.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Export — global panel for the latest run
# ---------------------------------------------------------------------------
with tab_export:
    st.header("Export latest run")
    st.caption("Downloads metrics (CSV/JSON), telemetry CSV, and an interactive "
               "Plotly HTML for sharing — all from the most recent benchmark or test.")
    _export_panel("latest")
