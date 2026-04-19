# ADR-001: Use FlashAttention over Vanilla Attention

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

Standard scaled dot-product attention materialises the full $N \times N$ attention matrix in GPU HBM (high-bandwidth memory). For $N=4096$ with 12 heads on A100:

$$\text{Memory} = 12 \times 4096^2 \times 2\text{B} = 402\text{MB}$$

This completely dominates GPU memory at long context lengths. Worse, it is **IO-bound** — the matrix is written to HBM and immediately read back, wasting bandwidth.

## Decision

We use `torch.nn.functional.scaled_dot_product_attention` (SDPA) as the default backend, which routes to **FlashAttention-v2/v3** on CUDA.

For maximum control and custom backward passes, a custom **Triton kernel** is available in `src/kernels/flash_attention.py`.

## Consequences

**Positive:**
- $4\times$ reduction in HBM IO → proportional throughput improvement
- $O(N)$ memory → enables context lengths of 128K+ without OOM
- No accuracy loss (mathematically equivalent to standard attention)

**Negative:**
- Triton dependency (Linux/CUDA only — Windows fallback to SDPA)
- Backward kernel is more complex to implement and debug
- Custom kernel requires Triton ≥ 2.1

## Alternatives Considered

| Option | Memory | Speed | Complexity |
|--------|--------|-------|------------|
| Vanilla PyTorch | O(N²) | 1× | Low |
| xFormers memory-efficient | O(N) | 1.8× | Medium |
| **FlashAttention (chosen)** | **O(N)** | **2.1×** | **Medium** |
| SAGE Attention | O(N) | 2.4× | High |

## References
- Dao et al. (2022) [FlashAttention](https://arxiv.org/abs/2205.14135)
- Dao et al. (2023) [FlashAttention-2](https://arxiv.org/abs/2307.08691)
