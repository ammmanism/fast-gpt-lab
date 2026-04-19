"""
Quantization Kernels — fast-gpt-lab
Provides INT8 and FP4 weight quantization hooks for deployment.
"""
import torch

def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric INT8 per-tensor quantization.
    Reduces memory footprint by 2x-4x for edge deployment.
    """
    scale = weight.abs().max() / 127.0
    q_weight = torch.round(weight / scale).to(torch.int8)
    return q_weight, scale

def dequantize_int8(q_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q_weight.to(torch.float32) * scale

class QuantizedLinear(torch.nn.Module):
    """
    Linear layer that stores weights in INT8 but computes in FP16/BF16.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.q_weight = torch.nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w = dequantize_int8(self.q_weight, self.scale).to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)
