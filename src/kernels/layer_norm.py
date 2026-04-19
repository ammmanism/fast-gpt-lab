"""
Fused LayerNorm Triton Kernel
Fuses normalization and weight scaling into a single hardware pass.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = x_zm * rstd * w + b
    tl.store(Y + cols, y, mask=mask)

def layernorm_forward(x, weight, bias, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_fwd_kernel[(M,)](x, y, weight, bias, mean, rstd, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y
