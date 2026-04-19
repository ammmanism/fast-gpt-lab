"""
MFU (Model FLOP Utilization) Profiler — fast-gpt-lab
Tracks what fraction of peak hardware FLOPs we actually achieve.

Reference: PaLM paper (Chowdhery et al., 2022) §2.4 — MFU definition
Formula:
    MFU = (observed_throughput × model_flops_per_token) / peak_hardware_flops
"""
import time
import math
import torch
from dataclasses import dataclass
from typing import Optional


# ─── Hardware peak FLOPs database ────────────────────────────────────────────
# Source: NVIDIA product specifications
PEAK_TFLOPS: dict[str, float] = {
    "A100-SXM-80GB":  312.0,   # bfloat16 tensor core
    "A100-PCIE-80GB": 312.0,
    "A100-SXM-40GB":  312.0,
    "H100-SXM":       989.4,   # H100 FP8 tensor core
    "H100-PCIE":      756.0,
    "RTX-4090":       330.3,
    "RTX-3090":       142.0,
    "RTX-3080":       119.0,
    "V100-SXM":       125.0,   # float16
}


@dataclass
class MFUReport:
    tokens_per_sec: float
    model_flops_per_token: int
    peak_tflops: float
    mfu: float              # 0..1 fraction
    step: int

    def __str__(self) -> str:
        return (
            f"[Step {self.step:6d}] "
            f"Throughput: {self.tokens_per_sec/1e3:6.1f}k tok/s | "
            f"Model FLOPs/token: {self.model_flops_per_token/1e9:.2f}B | "
            f"MFU: {self.mfu*100:.1f}%"
        )


class MFUProfiler:
    """
    Online MFU tracker for training runs.
    
    Usage:
        profiler = MFUProfiler(model, batch_size=12, seq_len=1024)
        profiler.start_step()
        # ... training step ...
        report = profiler.end_step(step_idx)
        print(report)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        seq_len: int,
        grad_accum_steps: int = 1,
        gpu_name: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.grad_accum_steps = grad_accum_steps
        self.tokens_per_step = batch_size * seq_len * grad_accum_steps

        self._flops_per_token = self._estimate_flops(model, seq_len)
        self._peak = self._get_peak_tflops(gpu_name)
        self._t_start: Optional[float] = None

        print(f"📊 MFUProfiler initialized")
        print(f"   Model FLOPs/token : {self._flops_per_token/1e9:.2f} GFLOPs")
        print(f"   Peak hardware     : {self._peak:.1f} TFLOPs")
        print(f"   Tokens/step       : {self.tokens_per_step:,}")

    def start_step(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t_start = time.perf_counter()

    def end_step(self, step: int) -> MFUReport:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - self._t_start

        tok_per_sec = self.tokens_per_step / t_elapsed
        achieved_tflops = (tok_per_sec * self._flops_per_token) / 1e12
        mfu = achieved_tflops / self._peak if self._peak > 0 else 0.0

        return MFUReport(
            tokens_per_sec=tok_per_sec,
            model_flops_per_token=self._flops_per_token,
            peak_tflops=self._peak,
            mfu=mfu,
            step=step,
        )

    @staticmethod
    def _estimate_flops(model: torch.nn.Module, seq_len: int) -> int:
        """
        Estimate FLOPs per token for a transformer model.
        
        Approximate formula (from PaLM paper):
            FLOPs ≈ 6 × N_params + 12 × n_layer × n_head × head_dim × seq_len
            
        The factor of 6 accounts for:
          - Forward pass:  2 × N_params  (GEMM: multiply + add)
          - Backward pass: 4 × N_params  (dL/dW + dL/dx)
          
        The second term captures attention's O(N²) cost.
        """
        total_params = sum(p.numel() for p in model.parameters())

        # Detect transformer config if available
        if hasattr(model, "config"):
            cfg = model.config
            n_layer = cfg.n_layer
            n_head  = cfg.n_head
            head_dim = cfg.n_embd // cfg.n_head
            attn_flops = 12 * n_layer * n_head * head_dim * seq_len
        else:
            attn_flops = 0

        return 6 * total_params + attn_flops

    @staticmethod
    def _get_peak_tflops(gpu_name: Optional[str] = None) -> float:
        if gpu_name and gpu_name in PEAK_TFLOPS:
            return PEAK_TFLOPS[gpu_name]
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            for key, flops in PEAK_TFLOPS.items():
                if key.split("-")[0] in name:
                    return flops
        return 1.0  # Avoid division by zero; MFU will be inflated


class MemoryAuditor:
    """
    Tracks GPU memory allocation, reserved, and peak across a training step.
    
    Usage:
        with MemoryAuditor("forward_pass") as audit:
            logits, loss = model(x, y)
        print(audit.summary())
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.stats: dict = {}

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.stats = {
                "allocated_gb":  torch.cuda.memory_allocated() / 1e9,
                "reserved_gb":   torch.cuda.memory_reserved() / 1e9,
                "peak_gb":       torch.cuda.max_memory_allocated() / 1e9,
            }
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000

    def summary(self) -> str:
        if not self.stats:
            return f"[{self.label}] No CUDA device"
        return (
            f"[{self.label}] "
            f"Alloc={self.stats['allocated_gb']:.2f}GB "
            f"Reserved={self.stats['reserved_gb']:.2f}GB "
            f"Peak={self.stats['peak_gb']:.2f}GB "
            f"Time={self.elapsed_ms:.1f}ms"
        )
