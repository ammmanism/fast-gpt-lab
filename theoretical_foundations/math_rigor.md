# Mathematical Foundations of fast-gpt-lab

This directory contains the formal derivations and proofs for the optimizations implemented in this project. Engineering without math is just guessing.

## Core Proofs

### 1. FlashAttention-v3 Gradient Stability
We derive the backward pass for the FP8 attention kernel, focusing on the preservation of the softmax normalization factor under low-precision accumulation.

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \dots $$

### 2. SwiGLU Fused Kernel Convergence
Proof of the numerical equivalence between the three-step PyTorch implementation and our single-pass Triton kernel.

### 3. FSDP Sharding Overhead
Calculation of the communication-to-compute ratio across varying GPU interconnect speeds (NVLink-4 vs. PCIe 5.0).

---

> "Pure math is the poetry of logic." — Our guiding principle for LLM architecture.
