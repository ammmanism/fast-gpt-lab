"""
Mixed Precision AMP Scaler — fast-gpt-lab
Handles deep neural network gradient scaling strictly for FP16,
and passes through seamlessly for BF16/FP32.
"""
import torch

class AMPScaler:
    """
    Robust gradient scaler wrapper for Automated Mixed Precision (AMP).
    Crucial for FP16 training to prevent vanishing gradients during backward pass
    due to limited exponent range.
    """
    def __init__(self, dtype: torch.dtype, enabled: bool = True):
        # GradScaler is only needed for FP16; BF16 has same range as FP32
        self.enabled = enabled and (dtype == torch.float16)
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss before backward pass."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscales gradients and calls optimizer step via scaler logic."""
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Updates the scale factor for the next iteration."""
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Explicitly unscale gradients before clipping."""
        if self.enabled:
            self.scaler.unscale_(optimizer)
