"""
KV Cache Implementation — fast-gpt-lab
Enables O(1) step time for auto-regressive decoding instead of O(T^2).
"""
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class KVCache:
    """
    Fixed-size pre-allocated KV cache for fast inference generation.
    """
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int = 0

    @classmethod
    def create(cls, batch_size: int, max_seq_len: int, n_head: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        k_cache = torch.zeros(batch_size, n_head, max_seq_len, head_dim, dtype=dtype, device=device)
        v_cache = torch.zeros(batch_size, n_head, max_seq_len, head_dim, dtype=dtype, device=device)
        return cls(k_cache, v_cache)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, pos: int):
        seq_len = k_new.size(2)
        self.k_cache[:, :, pos:pos+seq_len, :] = k_new
        self.v_cache[:, :, pos:pos+seq_len, :] = v_new
        self.seq_len = max(self.seq_len, pos + seq_len)
        
        return self.k_cache[:, :, :self.seq_len, :], self.v_cache[:, :, :self.seq_len, :]
    
    def reset(self):
        self.seq_len = 0
