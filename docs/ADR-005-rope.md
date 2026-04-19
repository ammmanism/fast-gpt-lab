# ADR-005: Rotary Positional Embeddings (RoPE)

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

Absolute positional embeddings (used in original Transformers and GPT-2) fail to generalize to sequence lengths longer than seen during training. ALiBi resolves this but degrades perplexity. Rotary Positional Embeddings (RoPE) provide relative positional information by rotating the query/key vectors in a 2D plane, preserving magnitude while injecting relative distance.

## Decision

We migrate from absolute learned positional embeddings (`wpe`) to **RoPE**.
We implement this via an in-place Triton kernel (`src/kernels/rotary.py`) to avoid the $O(B \times T \times D)$ memory spike caused by broadcasting sine and cosine frequencies in pure PyTorch.

## Consequences
- ✅ Better length extrapolation enabling infinite context windows.
- ✅ Eliminated the `wpe` matrix, saving ~10M parameters.
- ❌ Requires Triton dependency to be performant.
