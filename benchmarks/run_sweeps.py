"""
Hardware Sweeps — fast-gpt-lab
Automates throughput measurement across multiple dimension configurations.
"""
import torch
import itertools
from src.vanilla.config import GPTConfig
from src.vanilla.model import GPT
from benchmarks.benchmark_suite import BenchmarkSuite

def run_hardware_sweep(output_csv: str = "sweep_results.csv"):
    batch_sizes = [4, 8, 16, 32]
    context_lengths = [1024, 2048, 4096]
    
    suite = BenchmarkSuite(warmup=5, repeat=20)
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg).cuda()
    model.eval()
    
    print("📈 Commencing Hardware Benchmark Sweep...")
    
    for bsz, seq_len in itertools.product(batch_sizes, context_lengths):
        # We use empty caches specifically for strict un-polluted timing
        torch.cuda.empty_cache()
        x = torch.randint(0, cfg.vocab_size, (bsz, seq_len), device="cuda")
        
        test_name = f"Forward: B={bsz}, T={seq_len}"
        try:
            # We wrap the model call in a lambda so the benchmark suite can execute it cleanly
            suite.run(test_name, lambda: model(x))
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️  OOM at B={bsz}, T={seq_len} -> Skipping")
            torch.cuda.empty_cache()

    suite.save_csv(output_csv)
    print("✅ Sweep Complete. Generating Markdown overview...")
    print(suite.to_markdown())

if __name__ == "__main__":
    run_hardware_sweep()
