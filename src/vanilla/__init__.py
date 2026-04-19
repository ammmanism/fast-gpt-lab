"""
__init__.py — src/vanilla package
"""
from .config import GPTConfig
from .model import GPT
from .data import DataLoader

__all__ = ["GPTConfig", "GPT", "DataLoader"]
