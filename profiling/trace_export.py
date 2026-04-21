"""
Chrome Trace Exporter — fast-gpt-lab
Standardized serialization for profiling events to Chrome tracing formats.
"""
import os
from typing import Optional

class TraceExporter:
    """Ensures Profiler output is strictly mapped to actionable chrome://tracing bounds."""
    
    @staticmethod
    def get_handler(log_dir: str = "./profiling_logs"):
        """Returns the tensorboard trace handler ensuring directory persistence."""
        import torch.profiler
        os.makedirs(log_dir, exist_ok=True)
        return torch.profiler.tensorboard_trace_handler(log_dir)
        
    @staticmethod
    def get_schedule(wait: int = 1, warmup: int = 2, active: int = 3, repeat: int = 1):
        """
        Calculates optimal stepping schedules to prevent out-of-memory trace logs.
        Massive JSON dumps crash browsers. We isolate the 'active' window heavily.
        """
        import torch.profiler
        return torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        )
