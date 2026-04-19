# ADR-006: Auto-Regressive KV Caching

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

During inference generation (sampling new tokens autoregressively), evaluating the transformer involves computing self-attention over all previous tokens. Without caching, the computational complexity scales as $O(T^2)$ for decoding $T$ tokens.

## Decision

We introduce a stateful `KVCache` object passed through the forward method during inference. 
The KV Cache stores the key and value embeddings for all previously generated tokens, reducing step cost to $O(T)$, avoiding redundant re-computation of `x @ W_k` and `x @ W_v`.

## Consequences
- ✅ Text generation throughput increased by ~20x for long sequences.
- ❌ Code complexity increased; specific attention functions must handle `(1, D)` queries against `(T, D)` keys.
- ❌ VRAM consumption scales linearly with batch size and context length, requiring memory management via techniques like PagedAttention if scaled.
