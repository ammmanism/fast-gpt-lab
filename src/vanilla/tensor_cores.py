"""
Tensor Core Hardware Alignment — fast-gpt-lab
Validates architectural matrices for multiplier efficiency.
"""
import torch

def enforce_tensor_core_alignment(vocab_size: int, hidden_dim: int, head_dim: int) -> bool:
    """
    Nvidia Tensor Cores (Ampere/Hopper) require matrix dimensions to be multiples 
    of 8 (FP16/BF16) or 16/32 to actually be utilized.
    If your vocab size is 50,257 (GPT-2), PyTorch falls back to slow CUDA cores!
    We must pad dimensions to multiples of 8.
    """
    errors = []
    if vocab_size % 8 != 0:
        errors.append(f"Vocab Size {vocab_size} is not a multiple of 8. Matmuls will be unoptimized.")
    
    if hidden_dim % 8 != 0:
        errors.append(f"Hidden Dim {hidden_dim} is not a multiple of 8.")

    if head_dim % 8 != 0:
        errors.append(f"Head Dim {head_dim} is not a multiple of 8.")

    if errors:
        for err in errors:
            print(f"⚠️ TENSOR CORE WARNING: {err}")
        return False
        
    return True

def pad_vocab_size(original_vocab_size: int, alignment: int = 8) -> int:
    """Pads the vocabulary size to the nearest multiple for hardware efficiency."""
    remainder = original_vocab_size % alignment
    if remainder == 0:
        return original_vocab_size
    return original_vocab_size + (alignment - remainder)
