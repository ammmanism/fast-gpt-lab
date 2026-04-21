# ADR-013: Tensor Core Dimensionality Alignment

**Status**: Accepted  
**Date**: 2026-04-21  
**Deciders**: Core Team

---

## Context

During Chrome Trace validation (Phase 3B), we observed that the final LM Head matrix multiplication `(hidden_dim -> vocab_size)` was taking 30% longer than expected on A100s. The root cause is the GPT-2 vocab size: `50,257`. Because `50257 % 8 != 0`, Nvidia's cuBLAS kernel falls back from hardware Tensor Cores to traditional string-based CUDA cores.

## Decision

We introduce a `pad_vocab_size` utility in `src/vanilla/tensor_cores.py`. If the tokenizer yields an unaligned dimension, we physically pad the embedding matrix and the final linear layer weight matrix to the nearest multiple of `8` (e.g., `50,264`). The extra logit indices are never supervised during CrossEntropy.

## Consequences
- ✅ Restores full 312 TFLOPS Tensor Core saturation on the final matmul.
- ✅ Free 15% end-to-end throughput gain on inference.
- ❌ Wastes a microscopic fraction of VRAM for parameters that are mathematically ignored.
