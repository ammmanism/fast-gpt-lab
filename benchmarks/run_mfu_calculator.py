#!/usr/bin/env python3
"""
MFU Calculator — fast-gpt-lab
Calculates Model FLOPs Utilization for inference on A100 SXM4.

MFU = (2 * N_params * tokens_per_second) / peak_hardware_flops
Peak Hardware FLOPS for A100 BF16 = 312.5 TFLOPS
"""
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

# Import model from fast-gpt-lab
from src.vanilla.model import GPT, GPTConfig

# A100 SXM4 BF16 Peak FLOPS
A100_BF16_PEAK_TFLOPS = 312.5e12


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model: nn.Module, batch_size: int, seq_len: int, warmup: int = 10, iters: int = 50) -> dict:
    """Benchmark inference throughput."""
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    total_tokens = batch_size * seq_len * iters
    tokens_per_sec = total_tokens / elapsed

    return {
        "elapsed_sec": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "latency_per_token_ms": (elapsed / total_tokens) * 1000,
    }


def calculate_mfu(tokens_per_sec: float, n_params: int) -> float:
    """Calculate Model FLOPs Utilization (MFU) for inference."""
    # 2 * N_params * tokens_per_second (inference FLOPs)
    achieved_flops = 2 * n_params * tokens_per_sec
    mfu = achieved_flops / A100_BF16_PEAK_TFLOPS
    return mfu * 100  # percentage


def main():
    print("=" * 60)
    print("MFU Calculator — fast-gpt-lab")
    print("=" * 60)

    # Model configurations to test
    configs = [
        ("GPT-Small", GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, block_size=1024)),
        ("GPT-Medium", GPTConfig(vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, block_size=2048)),
        ("GPT-Large", GPTConfig(vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, block_size=4096)),
    ]

    results = []
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, config in configs:
        print(f"\n--- Benchmarking {name} ---")
        print(f"Config: layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}")

        model = GPT(config).cuda()
        model = model.to(memory_format=torch.channels_last)
        torch.compile(model, mode="reduce-overhead")

        n_params = count_parameters(model)
        print(f"Parameters: {n_params / 1e6:.2f}M")

        # Benchmark at different sequence lengths
        for seq_len in [512, 1024, 2048]:
            if seq_len > config.block_size:
                continue

            batch_size = max(1, 8192 // seq_len)  # Keep total tokens ~constant

            print(f"  seq_len={seq_len}, batch={batch_size}...")
            bench = benchmark_inference(model, batch_size, seq_len)

            mfu = calculate_mfu(bench["tokens_per_sec"], n_params)
            achieved_tflops = (2 * n_params * bench["tokens_per_sec"]) / 1e12

            result = {
                "model": name,
                "n_params_m": n_params / 1e6,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "tokens_per_sec": bench["tokens_per_sec"],
                "latency_per_token_ms": bench["latency_per_token_ms"],
                "achieved_tflops": achieved_tflops,
                "mfu_percent": mfu,
            }
            results.append(result)

            print(f"    Tokens/s: {bench['tokens_per_sec']:,.0f}")
            print(f"    Latency:  {bench['latency_per_token_ms']:.3f} ms/token")
            print(f"    Achieved: {achieved_tflops:.2f} TFLOPS")
            print(f"    MFU:      {mfu:.2f}%")

    # Save results
    output_json = results_dir / "mfu_report.json"
    output_txt = results_dir / "mfu_report.txt"

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(output_txt, "w") as f:
        f.write("MFU REPORT — fast-gpt-lab\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<15} {'Params(M)':>10} {'SeqLen':>8} {'Batch':>6} {'Tokens/s':>14} {'Lat(ms)':>10} {'TFLOPS':>10} {'MFU%':>8}\n")
        f.write("-" * 90 + "\n")
        for r in results:
            f.write(f"{r['model']:<15} {r['n_params_m']:>10.2f} {r['seq_len']:>8} {r['batch_size']:>6} {r['tokens_per_sec']:>14,.0f} {r['latency_per_token_ms']:>10.3f} {r['achieved_tflops']:>10.2f} {r['mfu_percent']:>8.2f}\n")

    print(f"\nResults saved to {output_json} and {output_txt}")
    print("=" * 60)


if __name__ == "__main__":
    main()