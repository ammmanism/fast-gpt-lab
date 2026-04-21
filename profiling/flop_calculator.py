"""
FLOP & MFU Calculator — fast-gpt-lab
Computes Model FLOP Utilization based on A100/H100 theoretical peaks.
"""
import torch
from dataclasses import dataclass
from src.vanilla.config import GPTConfig

@dataclass
class HardwareSpec:
    name: str
    peak_tflops: float # BF16/FP16 Tensor Core FLOPS
    memory_bw_gbps: float

# Peak hardware constants (without sparsity)
HARDWARE_DB = {
    "A100_SXM4_80GB": HardwareSpec(name="A100", peak_tflops=312.0, memory_bw_gbps=2039.0),
    "H100_SXM5_80GB": HardwareSpec(name="H100", peak_tflops=989.0, memory_bw_gbps=3350.0),
    "RTX_4090": HardwareSpec(name="RTX 4090", peak_tflops=165.2, memory_bw_gbps=1008.0)
}

def estimate_flops_per_token(cfg: GPTConfig) -> float:
    """
    Estimates the number of FLOPs required for a single forward pass per token.
    Uses the PaLM paper approximation: 6 * N + 12 * L * H * Q * T
    Simplified approximation for GPT architecture: ~ 2 * N (forward) + 4 * N (backward).
    Here we focus strictly on the Forward pass dense matmuls.
    """
    # N = total parameters
    return 2.0 * cfg.n_params

def calculate_mfu(cfg: GPTConfig, batch_size: int, seq_len: int, time_per_iter_ms: float, hw_name: str = "A100_SXM4_80GB") -> float:
    """
    Calculate Model FLOP Utilization (MFU).
    time_per_iter_ms: Time taken for one full Forward+Backward pass.
    """
    if hw_name not in HARDWARE_DB:
        raise ValueError(f"Hardware {hw_name} not in database.")
    
    hw = HARDWARE_DB[hw_name]
    
    # 6 FLOPs per param per token (2 for forward, 4 for backward approximations)
    flops_per_iter = 6.0 * cfg.n_params * batch_size * seq_len
    
    # Convert iter time to seconds
    iter_s = time_per_iter_ms / 1000.0
    
    # Achieved TFLOPS
    achieved_tflops = (flops_per_iter / iter_s) / 1e12
    
    # Ratio
    mfu = achieved_tflops / hw.peak_tflops
    return mfu
