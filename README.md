# GPU Egy – GPU Energy Benchmarks

GPU Egy is a local **GPU energy lab** you run on your own machine.

It uses **Streamlit + PyTorch + NVML** to benchmark your NVIDIA GPU and show:

- Compute performance – GFLOPs/s and **Joules per GFLOP**  
- Memory streaming – GB/s and **Joules per GB moved**  
- AI-style workloads – tokens/s and **Joules per token**  
- Rule-based **energy advice** to suggest ways to reduce energy for the same work

No cloud, no external AI calls: everything runs locally on your GPU.

---

## 1. What you need

Before installing:

- NVIDIA GPU with recent drivers  
  - `nvidia-smi` should work in a terminal  
- CUDA-enabled PyTorch  
  - `python -c "import torch; print(torch.cuda.is_available())"` → should print `True`  
- Python 3.9 or newer  
- Linux or Windows (macOS is not supported for CUDA)

---

## 2. Quick start (recommended)

Open a terminal and run:

```bash
# 1) Get the code
git clone https://github.com/<YOUR_GITHUB_USERNAME>/gpuegy.git
cd gpuegy

# 2) Create & activate a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the app
streamlit run streamlitapp.py
```

Then open the URL shown in the terminal (usually):

- http://localhost:8501

You are now running GPU Egy locally on **your GPU**.

---

## 3. Features

### GPU info

- Shows GPU name, driver/CUDA version  
- Current temperature, power, and memory usage

### Matmul benchmark (compute)

- Runs matrix multiply benchmarks on your GPU  
- Reports:
  - GFLOPs/s (throughput)  
  - average power and total energy (J)  
  - **energy per GFLOP (J/GFLOP)**  
- Lets you vary matrix sizes and repetitions

### Memory bandwidth benchmark

- Streams a large tensor in GPU memory (read/write loop)  
- Reports:
  - effective bandwidth (GB/s)  
  - average power and total energy (J)  
  - **energy per GB moved (J/GB)**  
- Helps you see the “energy cost of moving data”

### AI presets (tiny Transformer block)

- Runs a small Transformer-like block with different:
  - batch sizes  
  - sequence lengths  
  - precisions (FP32 / FP16 / BF16)  
- Reports:
  - tokens processed and tokens/s  
  - average power and total energy (J)  
  - **energy per token (J/token)**  
  - memory usage and utilization  
- Lets you explore:
  - batch size vs throughput vs energy  
  - long vs short sequences  
  - precision vs energy efficiency

### Energy test suite (pytest feel)

- A small registry of tests (matmul, memory, AI presets) with thresholds, e.g.:
  - minimum GFLOPs/s  
  - maximum energy per token / per GB / per test  
- Runs tests and marks them as PASS/FAIL  
- Useful to check if changes (driver, batch size, precision, power profile) improve or worsen energy efficiency

### History & analysis

- Logs each run into `benchmark_results.csv`  
- History tab lets you:
  - view past runs  
  - filter by benchmark type or test name  
  - visualize energy vs latency, energy per token vs tokens/s, etc.

### Rule-based energy advice (local)

- After each run the app can show short suggestions based on metrics, e.g.:
  - “GPU underutilized – try larger batch size to reduce J/token.”  
  - “Memory-bound – high mem util, moderate GPU util; consider shorter sequence or quantization.”  
  - “FP16 uses less energy per token than FP32 for this preset.”

No external AI services are called; this logic is purely local.

---

## 4. Files overview

- `streamlitapp.py`  
  Main Streamlit app with tabs:
  - GPU info  
  - Matmul benchmark  
  - Memory bandwidth  
  - AI presets  
  - Test suite  
  - History  
  - Telemetry viewer

- `AIpoweredmemory.py`  
  - Defines the tiny Transformer block  
  - `AI_PRESETS` (batch/seq/precision configurations)  
  - `run_ai_preset(preset_name: str)` to run one preset and return metrics

- `memory.py`  
  - `run_memory_bandwidth_benchmark(num_bytes, passes, ...)`  
  - Uses NVML to measure power while streaming a large tensor

- `requirements.txt`  
  Python dependencies for the app

---

## 5. Typical things to try

Once the app is running:

1. **Matmul: energy per GFLOP**
   - Run different matrix sizes (e.g. 1024, 2048, 4096)  
   - See how GFLOPs/s and J/GFLOP change  
   - Compare across GPUs or driver versions

2. **Memory: energy per GB moved**
   - Try 1 GB vs 4 GB tensors  
   - Increase passes from 20 to 100  
   - See if your GPU is more energy-efficient at certain sizes

3. **AI presets: batch size and precision**
   - Compare FP32 vs FP16 at the same batch size and sequence length  
   - Compare batch=8 vs batch=32; watch tokens/s and J/token  
   - See how longer sequences (e.g. seq=512) affect memory utilization and J/token

4. **Test suite**
   - Run the full test suite and see which tests pass your thresholds  
   - Change batch size/precision, rerun, and see if energy per work unit improves

---

## 6. Troubleshooting

- `torch.cuda.is_available()` is False  
  - Install a CUDA-enabled PyTorch build  
  - Check your NVIDIA driver and CUDA toolkit

- `pynvml.NVMLError` or power/temp are missing  
  - Ensure `nvidia-smi` works  
  - You may need to install `nvidia-ml-py` / `pynvml` and run as a user with permission to query NVML

- Streamlit app runs but some benchmarks are empty  
  - Check that `torch` sees your GPU (`torch.cuda.device_count()` > 0)  
  - Check logs in the console running `streamlitapp.py`

---

## 7. License & contributions

(Choose a license you like, e.g. MIT, Apache-2.0.)

Contributions are welcome:

- new benchmarks (kernels, models)  
- better energy advice rules  
- deeper history analysis or visualization

---

## 8. Website: gpuegy.dev

Visit **https://gpuegy.dev/** for:

- a quick overview of GPU Egy  
- screenshots and examples  
- direct links to this repo and usage instructions

The app always runs **on your GPU**, from your machine. Nothing is sent to a server unless you choose to share results yourself.
