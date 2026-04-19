# Mathematical Foundations — SwiGLU

## 1. GLU Family

Gated Linear Units (Dauphin et al., 2017) are defined as:

$$
\text{GLU}(x, W, V, b, c) = (xW + b) \otimes \sigma(xV + c)
$$

where $\sigma$ is an activation function and $\otimes$ is element-wise multiplication.

---

## 2. SwiGLU Derivation

SwiGLU (Shazeer, 2020) uses Swish / SiLU as the gate activation:

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

$$
\text{SwiGLU}(x) = \text{Swish}(xW_\text{gate}) \otimes (xW_\text{up})
$$

**Full FFN**:

$$
\text{FFN}_\text{SwiGLU}(x) = \text{SwiGLU}(x, W_\text{gate}, W_\text{up}) \cdot W_\text{down}
$$

---

## 3. Why SwiGLU > GELU

| Property | GELU | SwiGLU |
|----------|------|--------|
| Gating mechanism | Fixed activation | Learned gate |
| Parameter efficiency | 2 matrices | 3 matrices (scaled by 2/3) |
| Gradient flow | Standard | Gated → selective |
| Used in | GPT-2, BERT | LLaMA, PaLM, Mistral, Gemma |

**Empirically**: SwiGLU consistently outperforms GELU by ~1-2 perplexity points at equivalent parameter count across LLaMA ablation studies.

---

## 4. Hidden Dimension Adjustment

To maintain parameter count parity with a 4× GELU FFN:

$$
d_{\text{ffn}}^{\text{SwiGLU}} = \frac{2}{3} \times 4 d = \frac{8d}{3}
$$

Then round to nearest multiple of 64 for tensor core alignment:

$$
d_{\text{ffn}} = 64 \cdot \left\lceil \frac{8d/3}{64} \right\rceil
$$

For $d=768$: $8 \times 768 / 3 = 2048$, already aligned.

---

## 5. SwiGLU Gradient

Let $g = Wx_\text{gate}$, $u = Wx_\text{up}$. Forward: $h = \sigma(g) \cdot u$.

Backward:

$$
\frac{\partial \ell}{\partial g} = \frac{\partial \ell}{\partial h} \otimes u \otimes \frac{\partial \text{Swish}(g)}{\partial g}
$$

$$
\frac{\partial \text{Swish}(g)}{\partial g} = \sigma(g) + g \cdot \sigma(g)(1 - \sigma(g)) = \sigma(g)(1 + g(1-\sigma(g)))
$$

$$
\frac{\partial \ell}{\partial u} = \frac{\partial \ell}{\partial h} \otimes \text{Swish}(g)
$$

This is implemented exactly in our fused Triton kernel in `src/kernels/swiglu.py`.

---

## 6. FP8 Stability Concern

In FP8 (E4M3 format), the dynamic range is $[2^{-6}, 448]$.

SiliU gates are bounded: $\text{Swish}(x) \in (-0.278, +\infty)$, creating asymmetric FP8 range usage.
Gate values concentrated near 0 lose precision.

**Mitigation**: Scale gate output by learned scalar $\alpha$ (per-channel scaling):
$$h = \alpha \cdot \text{Swish}(g) \otimes u$$
This is the approach used in Transformer Engine (NVIDIA) FP8 SwiGLU.
