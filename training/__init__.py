"""
__init__.py — training package
"""
from .ddp import ddp_setup, ddp_teardown, wrap_ddp
from .fsdp import setup_distributed, teardown_distributed, wrap_model_fsdp
from .grad_checkpoint import apply_gradient_checkpointing

__all__ = [
    "wrap_model_fsdp", "setup_distributed", "teardown_distributed",
    "wrap_ddp", "ddp_setup", "ddp_teardown",
    "apply_gradient_checkpointing",
]
