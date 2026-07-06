"""
__init__.py — eval package
"""
from .hellaswag import evaluate_hellaswag
from .perplexity import evaluate_perplexity

__all__ = ["evaluate_perplexity", "evaluate_hellaswag"]
