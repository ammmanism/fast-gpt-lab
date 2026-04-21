# ADR-011: Model FLOP Utilization (MFU) Standardization

**Status**: Accepted  
**Date**: 2026-04-21  
**Deciders**: Core Team

---

## Context

"Tokens per second" is a highly deceptive metric because it scales quadratically with parameter count and sequence length. We need an objective, hardware-agnostic metric to prove our custom Triton kernels are actually extracting maximum potential from our Silicon (A100/H100).

## Decision

We standardize on **Model FLOP Utilization (MFU)**. We define the baseline equation as $6N$ FLOPs per token (representing the dense matmuls during the forward and backward pass blocks). The MFU ratio determines the percentage of theoretical peak hardware performance achieved by our implementation. MFU >= 55% is our minimum acceptable quality boundary for scaling.

## Consequences
- ✅ Allows true comparative analysis across A100s, H100s, and 4090s.
- ✅ Creates a rigorous, scientific threshold for merging performance PRs.
- ❌ Does not penalize parameter sparsity or activation memory overhead; it measures purely raw arithmetic layout efficiency.
