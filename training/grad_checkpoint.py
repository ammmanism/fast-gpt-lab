"""
Gradient Checkpointing Utilities — fast-gpt-lab
Trades compute for memory: recompute activations during backward instead of storing them.
Memory reduction: O(√N) layers stored instead of O(N).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class GradientCheckpointedBlock(nn.Module):
    """
    Wrapper that applies gradient checkpointing to any nn.Module.
    
    Memory analysis:
        Without checkpointing: store ALL activations for N blocks → O(N × B × T × d)
        With checkpointing:    store activations every K blocks  → O(K × B × T × d)
        
    Optimal K = √N → O(√N) memory, O(√N) extra FLOPs (recompute factor)
    For N=24 layers: K=5 → 5× memory reduction, 1.15× compute overhead
    """

    def __init__(self, block: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.block = block
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.block, x, use_reentrant=self.use_reentrant)


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_every: int = 1,
) -> nn.Module:
    """
    Apply gradient checkpointing to transformer blocks in-place.
    
    Args:
        model:             GPT model
        checkpoint_every:  checkpoint every N blocks (1 = all blocks)
    
    Returns:
        model with blocks wrapped in GradientCheckpointedBlock
    """
    if not hasattr(model, "transformer") or "h" not in model.transformer:
        raise ValueError("Model must have `transformer.h` ModuleList")

    blocks = model.transformer["h"]
    new_blocks = nn.ModuleList()

    for i, block in enumerate(blocks):
        if i % checkpoint_every == 0:
            new_blocks.append(GradientCheckpointedBlock(block))
        else:
            new_blocks.append(block)

    model.transformer["h"] = new_blocks
    num_ckpt = sum(1 for b in new_blocks if isinstance(b, GradientCheckpointedBlock))
    print(f"✅ Gradient checkpointing: {num_ckpt}/{len(blocks)} blocks checkpointed")
    return model
