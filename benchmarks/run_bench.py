"""
Benchmark Runner — fast-gpt-lab
Compares fast-gpt-lab throughput vs NanoGPT-style baseline and PyTorch vanilla.
"""
import time
import torch
import argparse
from dataclasses import dataclass


@dataclass
class BenchResult:
    name: str
    tokens_per_sec: float
    memory_gb: float
    loss: float
    mfu_pct: float


def setup_model(variant: str, device: str) -> tuple:
    """Return (model, batch, seq_len) for each variant."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.vanilla.config import GPTConfig
    from src.vanilla.model import GPT

    cfg = GPTConfig.gpt2_small()
    if variant == "swiglu":
        cfg.mlp_variant = "swiglu"
    else:
        cfg.mlp_variant = "gelu"

    model = GPT(cfg).to(device)
    if device == "cuda":
        model = model.half()  # fp16 for fair comparison
    B, T = 8, 512
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    return model, x, y, B * T


def run_benchmark(variant: str, device: str, steps: int = 20) -> BenchResult:
    model, x, y, tokens = setup_model(variant, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(5):
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        last_loss = loss.item()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    tok_per_sec = tokens * steps / elapsed
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    # Estimate MFU
    total_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * total_params
    peak_tflops = 312.0  # A100
    achieved_tflops = tok_per_sec * flops_per_token / 1e12
    mfu_pct = achieved_tflops / peak_tflops * 100

    return BenchResult(
        name=variant,
        tokens_per_sec=tok_per_sec,
        memory_gb=mem_gb,
        loss=last_loss,
        mfu_pct=mfu_pct,
    )


def print_table(results: list[BenchResult]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Implementation':<20} {'Tok/s':>10} {'MFU':>8} {'Memory':>10} {'Loss':>8}")
    print("-" * 72)
    baseline_tps = results[0].tokens_per_sec if results else 1.0
    for r in results:
        speedup = r.tokens_per_sec / baseline_tps
        print(
            f"{r.name:<20} "
            f"{r.tokens_per_sec/1e3:>8.1f}k "
            f"{r.mfu_pct:>7.1f}% "
            f"{r.memory_gb:>8.2f}GB "
            f"{r.loss:>8.4f}  "
            f"(×{speedup:.2f})"
        )
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="fast-gpt-lab benchmark suite")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    print(f"🚀 fast-gpt-lab Benchmark Suite")
    print(f"   Device: {args.device.upper()}")
    if args.device == "cuda":
        print(f"   GPU   : {torch.cuda.get_device_name(0)}")

    variants = ["gelu", "swiglu"]
    results = [run_benchmark(v, args.device, args.steps) for v in variants]
    print_table(results)


if __name__ == "__main__":
    main()
