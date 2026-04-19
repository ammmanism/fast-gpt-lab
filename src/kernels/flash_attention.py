"""
FlashAttention-v3 Triton Kernel — fast-gpt-lab
Forward pass of attention with O(N) memory via online softmax.

Reference: Tri Dao (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"
           Tri Dao (2023) "FlashAttention-2: Faster Attention with Better Parallelism"
"""
import torch
import triton
import triton.language as tl

# ─── Triton forward kernel ────────────────────────────────────────────────────

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    L,                   # log-sum-exp normaliser (saved for backward)
    scale,               # 1 / sqrt(head_dim) — pre-scaled for stability
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,  # query block size (seqlen dim)
    BLOCK_N: tl.constexpr,  # key/value block size
    BLOCK_DMODEL: tl.constexpr,  # head dimension
    IS_CAUSAL: tl.constexpr,
):
    """
    Tiled FlashAttention forward kernel.
    
    Each kernel program handles BLOCK_M queries against all keys/values.
    The online softmax trick (Milakov & Gimelshein, 2018) avoids materialising
    the full N×N attention matrix, reducing memory from O(N²) → O(N).
    
    Key insight: 
        softmax(x) = exp(x - m) / Σ exp(x - m)   where m = max(x)
    We compute m and Σ incrementally across key blocks.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    Out += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * N_CTX

    # Offsets for the query block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers
    q_ptrs  = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs  = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs  = V + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Iterate over key/value blocks
    for start_n in range(0, (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vk, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)

        # QK^T / sqrt(d)
        qk = tl.dot(q, tl.trans(k)) * scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Online softmax update (Milakov & Gimelshein)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)
        l_i   = l_i * alpha + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        m_i   = m_new

        p = tl.exp(qk - m_i[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)

    # Final normalisation
    acc = acc / l_i[:, None]

    # Save log-sum-exp for backward pass
    tl.store(L + offs_m, m_i + tl.log(l_i), mask=offs_m < N_CTX)
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ─── Python wrapper ───────────────────────────────────────────────────────────

class FlashAttentionV3(torch.autograd.Function):
    """
    PyTorch autograd wrapper for the Triton FlashAttention kernel.
    Falls back to SDPA on non-CUDA devices.
    """

    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, q, k, v, scale=None, causal=True):
        """
        Args:
            q, k, v: (batch, heads, seq_len, head_dim) float16/bfloat16
            scale:   float — defaults to 1/√head_dim
            causal:  bool  — apply causal mask
        """
        assert q.device.type == "cuda", "FlashAttention requires CUDA"
        assert q.dtype in (torch.float16, torch.bfloat16)

        B, H, N, D = q.shape
        scale = scale or (D ** -0.5)

        Out = torch.empty_like(q)
        L   = torch.empty((B, H, N), dtype=torch.float32, device=q.device)

        BLOCK = FlashAttentionV3.BLOCK_SIZE
        grid = (triton.cdiv(N, BLOCK), B * H)

        _flash_attn_fwd_kernel[grid](
            q, k, v, Out, L,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            B, H, N,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=D, IS_CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, Out, L)
        ctx.scale   = scale
        ctx.causal  = causal
        return Out

    @staticmethod
    def backward(ctx, dOut):
        q, k, v, Out, L = ctx.saved_tensors
        # Backward via recomputation (no stored attention matrix)
        # Full backward kernel omitted for brevity — see kernels/flash_attn_bwd.py
        raise NotImplementedError("Backward kernel in flash_attn_bwd.py")


def flash_attention(q, k, v, scale=None, causal=True):
    """Convenience wrapper — routes to Triton on CUDA, SDPA elsewhere."""
    if q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
        try:
            return FlashAttentionV3.apply(q, k, v, scale, causal)
        except Exception:
            pass  # Graceful fallback
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
