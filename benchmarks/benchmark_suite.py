"""
Benchmark Utilities — fast-gpt-lab
High-precision timers and throughput calculators for hardware-aware profiling.
"""
import torch
import time
import pandas as pd
from typing import Callable, List, Dict

class BenchmarkSuite:
    """
    Standardized suite for measuring kernel performance.
    Handles warmups, synchronization, and statistical aggregation.
    """
    def __init__(self, warmup: int = 10, repeat: int = 50):
        self.warmup = warmup
        self.repeat = repeat
        self.results = []

    def run(self, name: str, func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a specific function."""
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.repeat):
            func(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        ms = start_event.elapsed_time(end_event) / self.repeat
        
        self.results.append({"kernel": name, "time_ms": ms})
        print(f"⏱️ {name:<25}: {ms:>8.3f} ms")
        return ms

    def save_csv(self, path: str):
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        print(f"📊 Benchmarks saved to {path}")

    def to_markdown(self) -> str:
        df = pd.DataFrame(self.results)
        return df.to_markdown(index=False)
