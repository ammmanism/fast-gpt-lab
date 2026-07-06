"""
__init__.py — src/kernels package
"""
from .flash_attention import FlashAttentionV3, flash_attention
from .fp8_utils import FP8Context, is_fp8_available
from .swiglu import FusedSwiGLULinear, fused_swiglu

__all__ = [
    "flash_attention", "FlashAttentionV3",
    "fused_swiglu", "FusedSwiGLULinear",
    "FP8Context", "is_fp8_available",
]
