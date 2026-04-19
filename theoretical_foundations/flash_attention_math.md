# Mathematical Foundations — FlashAttention

## 1. Standard Attention (Quadratic Memory Bottleneck)

Given queries $\mathbf{Q} \in \mathbb{R}^{N\times d}$, keys $\mathbf{K} \in \mathbb{R}^{N\times d}$, values $\mathbf{V} \in \mathbb{R}^{N\times d}$:

$$
\mathbf{O} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

**Problem**: materialising $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N\times N}$ requires $O(N^2)$ memory.
For $N=8192, d=128, \text{fp16}$: $8192^2 \times 128 \times 2\text{B} = \mathbf{17\text{ GB}}$ per head.

---

## 2. Online Softmax (Milakov & Gimelshein, 2018)

The key identity enabling FlashAttention:

$$
\text{softmax}(x_1, \ldots, x_n) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

We can compute this **incrementally** over blocks. For two partial results with statistics $(m_1, \ell_1)$ and $(m_2, \ell_2)$:

$$
m_{\text{new}} = \max(m_1, m_2)
$$
$$
\ell_{\text{new}} = e^{m_1 - m_{\text{new}}} \cdot \ell_1 + e^{m_2 - m_{\text{new}}} \cdot \ell_2
$$
$$
\mathbf{O}_{\text{new}} = \frac{\ell_1 \cdot e^{m_1 - m_{\text{new}}} \cdot \mathbf{O}_1 + \ell_2 \cdot e^{m_2 - m_{\text{new}}} \cdot \mathbf{O}_2}{\ell_{\text{new}}}
$$

This is the **core trick** in our Triton kernel (`src/kernels/flash_attention.py`).

---

## 3. IO Complexity Analysis

| Algorithm | HBM Reads/Writes | Memory |
|-----------|-----------------|--------|
| Standard Attention | $O(N^2 d)$ | $O(N^2)$ |
| FlashAttention | $O(N^2 d / M)$ | $O(N)$ |

where $M$ = SRAM size (typically 20MB on A100).

**Speedup**: FlashAttention is IO-bound, not compute-bound. Reducing HBM traffic by $4\times$ (from not writing the attention matrix) yields proportional wall-clock speedup.

---

## 4. Gradient of Scaled Dot-Product Attention

Let $\mathbf{P} = \text{softmax}(\mathbf{S}/\sqrt{d})$, $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$.

Given upstream gradient $d\mathbf{O}$:

$$
d\mathbf{V} = \mathbf{P}^\top d\mathbf{O}
$$
$$
d\mathbf{P} = d\mathbf{O} \cdot \mathbf{V}^\top
$$
$$
d\mathbf{S} = \mathbf{P} \odot \left(d\mathbf{P} - \sum_j d\mathbf{P}_{ij} \mathbf{P}_{ij}\right) / \sqrt{d}
$$
$$
d\mathbf{Q} = d\mathbf{S} \cdot \mathbf{K}, \quad d\mathbf{K} = d\mathbf{S}^\top \cdot \mathbf{Q}
$$

**FlashAttention backward trick**: Instead of storing $\mathbf{P} \in O(N^2)$, we store only the log-sum-exp $L_{ij} = m_i + \log \ell_i$ (the `L` tensor in our kernel), and **recompute** $\mathbf{P}$ during backward via $P_{ij} = e^{S_{ij} - L_{ij}}$.

---

## 5. Causal Masking

For auto-regressive LMs, we enforce $\mathbf{P}_{ij} = 0$ for $j > i$:

$$
\mathbf{S}_{ij} = \begin{cases} \mathbf{Q}_i \cdot \mathbf{K}_j / \sqrt{d} & j \leq i \\ -\infty & j > i \end{cases}
$$

In our Triton kernel, this is implemented as a tile-level conditional:
```python
if IS_CAUSAL:
    causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(causal_mask, qk, float("-inf"))
```

Tiles where all queries precede all keys are **skipped entirely** — saving O(N²/2) compute.
