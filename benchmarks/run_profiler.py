"""
PyTorch Profiler Orchestration — fast-gpt-lab
Hooks deeply into the forward/backward pass to generate Chrome Traces and Flop counts.
"""
import torch
import torch.profiler
from src.vanilla.config import GPTConfig
from src.vanilla.model import GPT
from profiling.trace_export import TraceExporter

def run_chrome_trace_profiling():
    print("🔬 Booting deep Chrome Trace instrumentation...")
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg).cuda()
    
    # Dummy data
    bsz, seq_len = 8, 1024
    x = torch.randint(0, cfg.vocab_size, (bsz, seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (bsz, seq_len), device="cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # We profile exactly 6 steps to capture the 'Active' window.
    schedule = TraceExporter.get_schedule()
    handler = TraceExporter.get_handler("./profiling_logs/trace_outputs")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(6):
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            prof.step()
            print(f"   Step {step} analyzed.")

    print("✅ Deep Profile Complete. Check ./profiling_logs for JSON Chrome Trace.")

if __name__ == "__main__":
    run_chrome_trace_profiling()
