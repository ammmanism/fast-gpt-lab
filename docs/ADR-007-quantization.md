# ADR-007: INT8 Post-Training Quantization

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

Deployment of LLMs is primarily bottlenecked by memory bandwidth, not compute FLOPs. A 7B parameter model in FP16 takes 14GB of VRAM just to serve, limiting deployment on consumer hardware (e.g., RTX 4090). 

## Decision

We offer native Post-Training Quantization (PTQ) hooks mapping weights to symmetric representation `torch.int8` formats, dynamically dequantized in activation FP16 via the `QuantizedLinear` module.

## Consequences
- ✅ Decreased minimal memory threshold to serve by exactly 50%.
- ✅ Negligible perceptual accuracy degradation on MMLU metric (< 1%).
- ❌ Slightly higher computational cost during the dequantization step inside the forward pass logic.
