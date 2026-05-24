from schemas import BenchmarkRun
from typing import List, Dict

CLOUD_COST_PER_HOUR = {
    "NVIDIA A100-SXM4-80GB": 3.00,
    "NVIDIA H100 80GB HBM3": 6.00,
    "NVIDIA GeForce RTX 4090": 0.70,
    "default": 1.50,
}

GPU_CATALOG = {
    "NVIDIA GeForce RTX 4090":    {"vram_gb": 24,  "tdp_w": 450, "mem_bw_gbs": 1008, "fp16_tflops": 165},
    "NVIDIA A100-SXM4-80GB":      {"vram_gb": 80,  "tdp_w": 400, "mem_bw_gbs": 2000, "fp16_tflops": 312},
    "NVIDIA H100 80GB HBM3":      {"vram_gb": 80,  "tdp_w": 700, "mem_bw_gbs": 3350, "fp16_tflops": 989},
    "NVIDIA GeForce RTX 3090":    {"vram_gb": 24,  "tdp_w": 350, "mem_bw_gbs": 936,  "fp16_tflops": 71},
    "NVIDIA GeForce RTX 3080":    {"vram_gb": 10,  "tdp_w": 320, "mem_bw_gbs": 760,  "fp16_tflops": 58},
}

def derive_metrics(run: BenchmarkRun) -> Dict:
    duration_s = run.energy_joules / max(run.mean_power_w, 1.0)
    total_samples = run.throughput_samples_per_s * duration_s
    joules_per_sample = run.energy_joules / max(total_samples, 1.0)
    throughput_per_watt = run.throughput_samples_per_s / max(run.mean_power_w, 1.0)

    hourly = CLOUD_COST_PER_HOUR.get(run.gpu_name, CLOUD_COST_PER_HOUR["default"])
    cost_to_target = (run.energy_joules / 3_600_000) * hourly  # rough energy-cost proxy

    memory_headroom_gb = run.vram_gb - run.peak_vram_gb
    memory_risk = 1.0 if run.oom_flag else max(0.0, 1.0 - memory_headroom_gb / 4.0)

    return {
        "run_id": run.run_id,
        "gpu_name": run.gpu_name,
        "batch_size": run.batch_size,
        "precision": run.precision,
        "joules_per_sample": joules_per_sample,
        "throughput_per_watt": throughput_per_watt,
        "throughput_samples_per_s": run.throughput_samples_per_s,
        "energy_joules": run.energy_joules,
        "cost_to_target_usd": cost_to_target,
        "memory_headroom_gb": memory_headroom_gb,
        "memory_risk_score": memory_risk,
        "oom_flag": run.oom_flag,
    }

def recommend(runs: List[BenchmarkRun], mode: str = "greenest") -> Dict:
    """
    mode: 'fastest' | 'greenest' | 'cheapest'
    Returns best GPU name + explanation.
    """
    feasible = [r for r in runs if not r.oom_flag]
    if not feasible:
        return {"winner": None, "reason": "All runs hit OOM."}

    scored = [derive_metrics(r) for r in feasible]

    key_map = {
        "fastest": "throughput_samples_per_s",
        "greenest": "joules_per_sample",
        "cheapest": "cost_to_target_usd",
    }
    key = key_map[mode]
    reverse = mode == "fastest"
    best = sorted(scored, key=lambda x: x[key], reverse=reverse)[0]

    if best["memory_headroom_gb"] < 2.0:
        warning = " ⚠️ Low memory headroom — consider smaller batch size."
    else:
        warning = ""

    reasons = {
        "fastest": f"highest throughput at {best['throughput_samples_per_s']:.0f} samples/s",
        "greenest": f"lowest energy at {best['joules_per_sample']:.4f} J/sample",
        "cheapest": f"lowest estimated cost at ${best['cost_to_target_usd']:.4f}",
    }

    return {
        "winner": best["gpu_name"],
        "mode": mode,
        "reason": f"{best['gpu_name']} wins ({reasons[mode]}, batch={best['batch_size']}, {best['precision']}){warning}",
        "scores": scored,
    }
