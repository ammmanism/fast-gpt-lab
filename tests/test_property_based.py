"""
Hypothesis suite for Triton FlashAttention kernel.
Comprehensive property-based testing for numerical correctness and memory safety.
"""
import torch
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import torch.nn.functional as F

from src.kernels.flash_attention import flash_attention

batch = st.integers(1, 8)
heads = st.integers(1, 16)
seq_len = st.integers(64, 8192).filter(lambda n: n % 64 == 0)
head_dim = st.sampled_from([64, 128])


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(b=batch, h=heads, n=seq_len, d=head_dim)
def test_fa_shape(b, h, n, d):
    torch.cuda.empty_cache()
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attention(q, k, v)
    assert out.shape == (b, h, n, d), f"shape mismatch {out.shape}"


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(b=batch, h=heads, n=seq_len, d=head_dim)
def test_fa_no_nan_inf(b, h, n, d):
    torch.cuda.empty_cache()
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attention(q, k, v)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(b=batch, h=heads, n=seq_len, d=head_dim)
def test_fa_parity(b, h, n, d):
    torch.cuda.empty_cache()
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    max_err = (out - ref).abs().max().item()
    assert max_err < 2e-2, f"parity failed: max_err={max_err:.4f} for ({b},{h},{n},{d})"


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(b=batch, h=heads, n=seq_len, d=head_dim)
def test_fa_determinism(b, h, n, d):
    """Same input must produce bitwise identical output across two runs."""
    torch.cuda.empty_cache()
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out1 = flash_attention(q, k, v)
    out2 = flash_attention(q, k, v)

    # Bitwise identical (allowing for non-determinism in SDPA fallback path)
    assert torch.allclose(out1, out2, rtol=0, atol=0), "Output not bitwise deterministic"


@settings(max_examples=1, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_fa_memory_scaling(data):
    """Test O(N) memory scaling: growth ratio < 20x for 8x seq_len increase."""
    b = data.draw(batch)
    h = data.draw(heads)
    d = data.draw(head_dim)

    seq_lens = [512, 1024, 2048, 4096]
    peak_memory = []

    for n in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        _ = flash_attention(q, k, v)
        torch.cuda.synchronize()

        peak_memory.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB

    # Check growth ratio: 4096 / 512 = 8x sequence length
    # Should be < 20x memory growth (linear scaling, not quadratic)
    growth_ratio = peak_memory[-1] / peak_memory[0]
    assert growth_ratio < 20, f"Memory growth ratio {growth_ratio:.2f}x exceeds 20x (quadratic detected)"

    # Additional check: memory should grow roughly linearly
    for i in range(1, len(seq_lens)):
        expected_ratio = seq_lens[i] / seq_lens[0]
        actual_ratio = peak_memory[i] / peak_memory[0]
        # Allow up to 3x overhead factor
        assert actual_ratio < expected_ratio * 3, f"Non-linear memory scaling at {seq_lens[i]}"