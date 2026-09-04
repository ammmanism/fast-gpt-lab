#!/usr/bin/env python3
"""
Micro-benchmark suite — fast-gpt-lab
Uses triton.testing.do_bench to measure per-kernel latency.
Compares FlashAttention-v3 vs PyTorch SDPA across sequence lengths.
"""
import json
import torch
import triton.testing
from pathlib import Path

from src.kernels.flash_attention import flash_attention


def benchmark_flash_attention():
    """Benchmark FlashAttention-v3 vs SDPA."""
    torch.manual_seed(42)
    device = "cuda"

    seq_lens = [1024, 2048, 4096, 8192]
    batch = 4
    heads = 32
    head_dim = 128  # Divisible by 8

    results = []

    for seq_len in seq_lens:
        print(f"\nBenchmarking seq_len={seq_len}...")

        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(5):
            _ = flash_attention(q, k, v)
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        # Benchmark custom kernel
        custom_ms = triton.testing.do_bench(
            lambda: flash_attention(q, k, v),
            warmup=10,
            rep=50,
        )

        # Benchmark PyTorch SDPA
        sdpa_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
            warmup=10,
            rep=50,
        )

        # Calculate TFLOPS
        # Attention FLOPs: 2 * B * H * N * N * D (QK^T) + 2 * B * H * N * N * D (PV) ≈ 4 * B * H * N^2 * D
        flops = 4 * batch * heads * seq_len * seq_len * head_dim

        custom_tflops = (flops / custom_ms / 1e9) if custom_ms > 0 else 0
        sdpa_tflops = (flops / sdpa_ms / 1e9) if sdpa_ms > 0 else 0

        speedup = sdpa_ms / custom_ms if custom_ms > 0 else 0

        result = {
            "seq_len": seq_len,
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "custom_kernel_ms": custom_ms,
            "sdpa_ms": sdpa_ms,
            "speedup": speedup,
            "custom_tflops": custom_tflops,
            "sdpa_tflops": sdpa_tflops,
            "flops": flops,
        }
        results.append(result)

        print(f"  Custom: {custom_ms:.3f} ms ({custom_tflops:.2f} TFLOPS)")
        print(f"  SDPA:   {sdpa_ms:.3f} ms ({sdpa_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def benchmark_swiglu():
    """Benchmark SwiGLU kernel."""
    from src.kernels.swiglu import fused_swiglu

    torch.manual_seed(42)
    device = "cuda"

    # Test different hidden dimensions
    configs = [
        (8192, 4096, 4096),   # M, K, N
        (4096, 8192, 8192),
        (2048, 16384, 16384),
    ]

    results = []

    for M, K, N in configs:
        print(f"\nBenchmarking SwiGLU M={M}, K={K}, N={N}...")

        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_gate = torch.randn(K, N, device=device, dtype=torch.bfloat16)
        w_up = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            _ = fused_swiglu(x, w_gate, w_up)
        torch.cuda.synchronize()

        custom_ms = triton.testing.do_bench(
            lambda: fused_swiglu(x, w_gate, w_up),
            warmup=10,
            rep=50,
        )

        # Reference: PyTorch eager
        ref_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.silu(x @ w_gate) * (x @ w_up),
            warmup=10,
            rep=50,
        )

        flops = 2 * M * K * N * 2  # 2 GEMMs
        custom_tflops = (flops / custom_ms / 1e9) if custom_ms > 0 else 0

        result = {
            "M": M, "K": K, "N": N,
            "custom_ms": custom_ms,
            "pytorch_ms": ref_ms,
            "speedup": ref_ms / custom_ms if custom_ms > 0 else 0,
            "custom_tflops": custom_tflops,
        }
        results.append(result)

        print(f"  Custom: {custom_ms:.3f} ms ({custom_tflops:.2f} TFLOPS)")
        print(f"  PyTorch: {ref_ms:.3f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")

    return results


def main():
    print("=" * 60)
    print("Micro-benchmark Suite — fast-gpt-lab")
    print("=" * 60)

    results = {
        "flash_attention": benchmark_flash_attention(),
        "swiglu": benchmark_swiglu(),
    }

    output_path = Path("benchmarks/results/micro_bench_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()