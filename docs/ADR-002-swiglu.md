# ADR-002: SwiGLU over GELU as Default FFN

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

The FFN in a transformer can use various activation functions. The GPT-2/3 family used GELU. Modern LLMs (LLaMA, PaLM, Mistral, Gemma) have converged on SwiGLU.

## Decision

Use **SwiGLU** as the default `mlp_variant` in `GPTConfig`. GELU is kept as an option for reproducibility with GPT-2 checkpoints.

## Reasoning

1. **Empirical superiority**: PaLM ablations show SwiGLU reduces perplexity by 0.4-1.8 points across model sizes vs GELU.
2. **Gating mechanism**: The learned gate $\sigma(xW_\text{gate})$ provides **selective neuron activation**, acting as a form of dynamic sparsity.
3. **Triton fusion**: The gate + up projection can be fused into a single kernel (`src/kernels/swiglu.py`), eliminating a memory round-trip.

## Parameter Count Adjustment

SwiGLU uses 3 weight matrices instead of 2. To match GELU FFN params:

$$d_{\text{ffn}}^{\text{SwiGLU}} = \frac{2}{3} \times 4d = \frac{8d}{3}$$

This is enforced in `SwiGLU.__init__`.

## Consequences

- ✅ Better downstream accuracy at equal training compute
- ✅ Fuseable Triton kernel available
- ❌ Cannot directly load GPT-2 GELU checkpoints (different weight shapes)
- ❌ 3 weight matrices = slightly harder to inspect/visualize

## References
- Shazeer (2020) [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Touvron et al. (2023) [LLaMA](https://arxiv.org/abs/2302.13971)
