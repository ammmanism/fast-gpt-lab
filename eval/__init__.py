"""
__init__.py — eval package
"""
from .perplexity import evaluate_perplexity
from .hellaswag import evaluate_hellaswag

__all__ = ["evaluate_perplexity", "evaluate_hellaswag"]
