# GPU Energy Bench

A `pytest`-style Streamlit app for measuring and reducing GPU energy.

## Install
```bash
pip install -r gpu_energy_bench/requirements.txt
```

## Run
```bash
streamlit run gpu_energy_bench/streamlit_app.py
```

## Layout
| File | Role |
|---|---|
| `nvml_utils.py`  | Robust NVML snapshot + power-limit control (extends `firstbasic.py`) |
| `telemetry.py`   | Threaded sampler + energy integration (extends `storedata.py`) |
| `kernels.py`     | Kernel registry. **Add new kernels here** with `register("name", fn)` |
| `registry.py`    | Test specs + threshold evaluation (PASS/FAIL) |
| `runner.py`      | Glue: kernel + telemetry → unified `RunMetrics` |
| `storage.py`     | SQLite store of every run for cross-config comparisons |
| `tests.yaml`     | **Edit this** to declare tests + thresholds |
| `streamlit_app.py` | UI (6 tabs: Info, Benchmark, Tests, Telemetry, Power, History) |

## Adding a kernel
```python
# in kernels.py
def _my_kernel(device, **params) -> KernelResult:
    # ... warmup, torch.cuda.synchronize(), time, count flops ...
    return KernelResult(elapsed_s=..., flops=..., repetitions=...)
register("my_kernel", _my_kernel)
```
Then reference it from `tests.yaml`.

## Energy-reduction experiments
1. **Power cap sweep**: in *Power limits*, set 150 W → 200 W → 250 W; rerun the same test each time; in *History*, the Time-vs-Energy scatter shows the Pareto front.
2. **Size sweep**: vary `size` in `tests.yaml`, look at *Energy per GFLOP vs size*.
3. **Precision sweep**: duplicate a test with `dtype: float16` / `bfloat16` and compare J/GFLOP.

## Notes
- `set_power_limit` shells out to `nvidia-smi -pl` and may need `sudo` (toggle in UI). If it fails, the app stays read-only.
- All NVML reads degrade to `N/A` instead of crashing (fan, p-state, power constraints often unsupported on consumer cards).
- Energy is computed by trapezoidal integration of sampled power; raise the telemetry rate in the sidebar for short kernels.
