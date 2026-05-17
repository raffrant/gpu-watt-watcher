# GPU Watt Watcher (GPU Egy)

GPU Watt Watcher is a small app you run on your own NVIDIA GPU.

It shows:
- How fast your GPU runs simple benchmarks (GFLOPs/s, tokens/s)
- How much energy those benchmarks use (Joules)
- Simple hints about what to change (batch size, sequence length, precision) to use less energy for the same work

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


## 3. Run it on your GPU

Requirements:
- NVIDIA GPU with drivers installed (`nvidia-smi` works)
- CUDA-enabled PyTorch
- Python 3.9+

Steps:

```bash
git clone https://github.com/raffrant/gpu-watt-watcher.git
cd gpu-watt-watcher
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cd gpu_energy_bench
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## 4. What you get

In the app you will find:

- **GPU info** – name, driver/CUDA version, current power, temperature, memory
- **Matmul benchmark** – GFLOPs/s and Joules per GFLOP
- **Memory benchmark** – GB/s and Joules per GB of data moved
- **AI presets** – tokens/s and Joules per token for a tiny Transformer block
- **Test suite** – a few predefined tests with pass/fail thresholds on energy and performance
- **History** – a CSV log of your runs so you can compare changes over time
---

## 5. License & contributions

(Choose a license you like, e.g. MIT, Apache-2.0.)

Contributions are welcome:

- new benchmarks (kernels, models)  
- better energy advice rules  
- deeper history analysis or visualization

---

## 6. Website: gpuegy.dev

Visit **https://gpuegy.dev/** for:

- a quick overview of GPU Egy  
- screenshots and examples  
- direct links to this repo and usage instructions

The app always runs **on your GPU**, from your machine. Nothing is sent to a server unless you choose to share results yourself.

## 7. Used Lovable to buy a domain and help with the web development. 
