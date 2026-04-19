"""
Chrome Trace Profiler — fast-gpt-lab
Generates JSON traces viewable in chrome://tracing for pipeline analysis.
"""
import json
import time
import torch
from contextlib import contextmanager
from pathlib import Path


class ChromeTracer:
    """
    Records CUDA kernel timing events and exports as Chrome JSON Trace.
    
    Usage:
        tracer = ChromeTracer()
        with tracer.record("forward_pass"):
            logits, loss = model(x, y)
        tracer.save("profiling/trace.json")
        
    Open in chrome://tracing to visualise pipeline.
    """

    def __init__(self):
        self.events: list[dict] = []
        self._pid = 0

    @contextmanager
    def record(self, name: str, tid: int = 0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter_ns()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter_ns()
            self.events.append({
                "name": name,
                "ph": "X",           # 'X' = complete event (has duration)
                "ts": t_start / 1e3,  # microseconds
                "dur": (t_end - t_start) / 1e3,
                "pid": self._pid,
                "tid": tid,
                "args": {},
            })

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"traceEvents": self.events}, f)
        print(f"📊 Chrome trace saved → {path}")
        print(f"   Open in: chrome://tracing  (or ui.perfetto.dev)")

    def summary(self) -> None:
        if not self.events:
            print("No events recorded.")
            return
        print(f"\n{'Event':<30} {'Duration':>12}")
        print("-" * 44)
        for e in sorted(self.events, key=lambda x: -x["dur"]):
            print(f"{e['name']:<30} {e['dur']/1e3:>10.2f}ms")
