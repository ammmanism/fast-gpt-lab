"""
Fused LayerNorm + Residual Add — fast-gpt-lab
Peak bandwidth optimization combining elementwise operations into a single kernel.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Residual,  # pointer to the residual block
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    if Residual is not None:
        Residual += row * stride

    # Compute mean
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    _var = tl.zeros([1], dtype=tl.float32)
    x_centered = tl.where(mask, x - mean, 0.)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    
    # Normalize and apply linear transformation
    x_hat = x_centered * rstd
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = x_hat * w + b
    
    # FUSED RESIDUAL BLOCK
    if Residual is not None:
        res = tl.load(Residual + cols, mask=mask)
        y = y + res
        
    tl.store(Y + cols, y, mask=mask)

class FusedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, residual=None, eps=1e-5):
        # Flatten x for Triton 1D grid distribution
        x_shape_og = x.shape
        x = x.view(-1, x_shape_og[-1])
        if residual is not None:
            residual = residual.view(-1, x_shape_og[-1])
            
        M, N = x.shape
        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8
        
        _layer_norm_fwd_fused[(M,)](
            x, y, weight, bias, residual, mean, rstd,
            x.stride(0), N, eps, 
            num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        
        return y.view(*x_shape_og)

def fused_layer_norm(x, weight, bias, residual=None, eps=1e-5):
    return FusedLayerNorm.apply(x, weight, bias, residual, eps)
