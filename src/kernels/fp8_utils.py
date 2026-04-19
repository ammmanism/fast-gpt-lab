"""
FP8 Mixed-Precision Utilities — fast-gpt-lab
Experimental support for FP8 training via NVIDIA Transformer Engine.

FP8 formats:
  E4M3: range [-448, 448], good for weights/activations (high precision near 0)
  E5M2: range [-57344, 57344], good for gradients (needs larger range)
"""
import torch
from typing import Optional


def is_fp8_available() -> bool:
    """Check if NVIDIA Transformer Engine FP8 is available."""
    try:
        import transformer_engine.pytorch as te
        return True
    except ImportError:
        return False


class FP8Context:
    """
    Context manager for FP8 training.
    
    Usage:
        with FP8Context(enabled=True):
            logits, loss = model(x, y)
    
    Falls back to bfloat16 if Transformer Engine not installed.
    """

    def __init__(self, enabled: bool = True, recipe_name: str = "delayed_scaling"):
        self.enabled = enabled and is_fp8_available()
        self.recipe_name = recipe_name
        self._ctx = None

        if enabled and not self.enabled:
            print("⚠️  FP8 requested but transformer-engine not installed — using bf16")

    def __enter__(self):
        if self.enabled:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling
            recipe = DelayedScaling(fp8_format="HYBRID", amax_history_len=32)
            self._ctx = te.fp8_autocast(enabled=True, fp8_recipe=recipe)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx is not None:
            self._ctx.__exit__(*args)


def estimate_fp8_savings(model: torch.nn.Module) -> dict[str, float]:
    """Estimate memory savings from FP8 quantization vs fp16."""
    total_params = sum(p.numel() for p in model.parameters())
    fp16_bytes = total_params * 2
    fp8_bytes  = total_params * 1  # FP8 = 1 byte per param
    return {
        "fp16_mb": fp16_bytes / 1e6,
        "fp8_mb":  fp8_bytes  / 1e6,
        "savings_mb": (fp16_bytes - fp8_bytes) / 1e6,
        "savings_pct": 50.0,
    }
