"""
__init__.py — training package
"""
from .fsdp import wrap_model_fsdp, setup_distributed, teardown_distributed
from .ddp import wrap_ddp, ddp_setup, ddp_teardown
from .grad_checkpoint import apply_gradient_checkpointing

__all__ = [
    "wrap_model_fsdp", "setup_distributed", "teardown_distributed",
    "wrap_ddp", "ddp_setup", "ddp_teardown",
    "apply_gradient_checkpointing",
]
