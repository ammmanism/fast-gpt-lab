"""
GPT Model Configuration — fast-gpt-lab
Reference: Attention Is All You Need (Vaswani et al., 2017)
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPTConfig:
    # ─── Architecture ──────────────────────────────────────────────────────────
    block_size: int = 1024          # Max sequence length (context window)
    vocab_size: int = 50304         # GPT-2 vocab padded to nearest multiple of 64
    n_layer: int = 12               # Number of transformer blocks
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Embedding dimension (d_model)
    n_kv_head: Optional[int] = None # For GQA; None = MHA (n_kv_head == n_head)

    # ─── MLP ───────────────────────────────────────────────────────────────────
    mlp_ratio: float = 4.0          # Hidden dim multiplier for FFN
    mlp_variant: str = "swiglu"     # "gelu" | "swiglu" — SwiGLU for performance

    # ─── Regularisation ────────────────────────────────────────────────────────
    dropout: float = 0.0            # Dropout (0.0 for inference / large models)
    attention_dropout: float = 0.0  # Separate attn dropout for ablations
    bias: bool = False              # No bias → faster, matches GPT-NeoX

    # ─── Initialisation ────────────────────────────────────────────────────────
    init_std: float = 0.02          # σ for N(0, σ²) weight init
    residual_scale: bool = True     # Scale residual proj by 1/√(2*n_layer)

    # ─── Precision ─────────────────────────────────────────────────────────────
    use_fp8: bool = False           # Experimental FP8 via transformer-engine
    use_flash_attn: bool = True     # Use FlashAttention-v2/v3 kernel

    # ─── Named presets ────────────────────────────────────────────────────────
    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        return cls(n_layer=36, n_head=20, n_embd=1280)

    @classmethod
    def gpt2_xl(cls) -> "GPTConfig":
        return cls(n_layer=48, n_head=25, n_embd=1600, vocab_size=50304)

    @classmethod
    def micro(cls) -> "GPTConfig":
        """Tiny model for unit tests — fits on CPU"""
        return cls(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=256)

    @property
    def n_params(self) -> int:
        """Rough parameter count (excl. embeddings)."""
        kv = self.n_kv_head or self.n_head
        attn = self.n_embd * (self.n_head + 2 * kv) * (self.n_embd // self.n_head)
        ffn_dim = int(self.mlp_ratio * self.n_embd)
        if self.mlp_variant == "swiglu":
            ffn_dim = int(ffn_dim * 2 / 3)
            mlp = 3 * self.n_embd * ffn_dim
        else:
            mlp = 2 * self.n_embd * ffn_dim
        return self.n_layer * (attn + mlp)

    def __repr__(self) -> str:
        return (
            f"GPTConfig({self.n_layer}L/{self.n_head}H/{self.n_embd}D "
            f"ctx={self.block_size} vocab={self.vocab_size} "
            f"~{self.n_params/1e6:.1f}M params)"
        )
