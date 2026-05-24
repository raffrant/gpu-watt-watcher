from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

@dataclass
class BenchmarkRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp_start: str = ""
    timestamp_end: str = ""
    hostname: str = ""
    gpu_name: str = ""
    gpu_uuid: str = ""
    vram_gb: float = 0.0
    driver_version: str = ""
    cuda_version: str = ""
    software_stack: str = "pytorch"
    software_version: str = ""
    model_name: str = ""
    workload_type: str = ""       # cnn_train, llm_finetune, inference, kernel
    dataset_name: str = ""
    precision: str = "fp32"       # fp32, fp16, bf16, fp8
    batch_size: int = 0
    grad_accum_steps: int = 1
    sequence_length: int = 0
    image_resolution: int = 0
    epochs_requested: int = 0
    epochs_completed: int = 0
    target_metric_name: str = ""
    target_metric_value: float = 0.0
    final_metric_value: float = 0.0
    throughput_samples_per_s: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    energy_joules: float = 0.0
    mean_gpu_util: float = 0.0
    mean_mem_util: float = 0.0
    peak_vram_gb: float = 0.0
    oom_flag: bool = False
    power_limit_w: float = 0.0
    estimated_cost_usd: float = 0.0

    def to_dict(self):
        return asdict(self)
