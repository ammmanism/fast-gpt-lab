"""
Vanilla GPT Model — fast-gpt-lab
Paper-to-code implementation that is both readable AND fast.
Uses torch.nn.functional.scaled_dot_product_attention (FlashAttention backend).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import GPTConfig


# ─── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Delegates to PyTorch's SDPA which routes to FlashAttention-v2/v3 on CUDA.
    
    Memory: O(N) with FlashAttention vs O(N²) naive — critical for long context.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.attention_dropout

        # Fused QKV projection — single matmul for 3x throughput
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        # ── QKV projection (fused) ─────────────────────────────────────────────
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, n_head, T, head_dim) for SDPA
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── Scaled Dot-Product Attention (FlashAttention backend) ─────────────
        dp = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        # ── Re-assemble heads ─────────────────────────────────────────────────
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ─── Feed-Forward Network ─────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    SwiGLU FFN: FFN_{SwiGLU}(x, W, V, W₂) = (Swish(xW) ⊙ xV)W₂
    Reference: Noam Shazeer, "GLU Variants Improve Transformer" (2020)
    
    Key property: gating selectively activates neurons → better gradient flow
    than GELU for large models. Used in LLaMA, PaLM, Mistral, Gemma.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # ⚠️  Must scale hidden_dim by 2/3 to keep param count ≈ 4x GELU MLP
        hidden_dim = int(config.mlp_ratio * config.n_embd * 2 / 3)
        # Round to nearest multiple of 64 for tensor core efficiency
        hidden_dim = 64 * math.ceil(hidden_dim / 64)

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) ⊙ up — the "gated" part that makes SwiGLU special
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELU_MLP(nn.Module):
    """Standard GPT-2 MLP with GELU — baseline comparison."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.n_embd)
        self.fc   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


def build_mlp(config: GPTConfig) -> nn.Module:
    if config.mlp_variant == "swiglu":
        return SwiGLU(config)
    elif config.mlp_variant == "gelu":
        return GELU_MLP(config)
    else:
        raise ValueError(f"Unknown mlp_variant: {config.mlp_variant!r}")


# ─── Transformer Block ────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    Pre-norm Transformer Block.
    Uses LayerNorm before attention/FFN (unlike original 'post-norm' GPT).
    Pre-norm dramatically improves training stability at scale.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = build_mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # residual + pre-norm attention
        x = x + self.mlp(self.ln_2(x))    # residual + pre-norm FFN
        return x


# ─── Full GPT Model ──────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    GPT Language Model — fast-gpt-lab implementation.
    
    Design decisions vs GPT-2 reference:
    ✓ Pre-norm (not post-norm) → better gradient flow
    ✓ Weight tying: wte.weight == lm_head.weight → -30M params free
    ✓ SwiGLU MLP by default → +2-3% accuracy on downstream tasks
    ✓ No bias in attention/FFN → matches GPT-NeoX & LLaMA convention
    ✓ sdpa() backend → FlashAttention-v2/v3 on CUDA automatically
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            "wpe":  nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            "drop": nn.Dropout(config.dropout),
            "h":    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # Rationale: vocab embedding & output projection share semantics.
        # Paper: Press & Wolf, 2017 — "Using Output Embedding to Improve LMs"
        self.transformer["wte"].weight = self.lm_head.weight

        # ── Weight initialisation ─────────────────────────────────────────────
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("c_proj.weight", "down_proj.weight")):
                # Special scaled init for residual projections (GPT-2 recipe)
                # σ = 0.02 / √(2 * n_layer): prevents residual stream blow-up
                std = config.init_std / math.sqrt(2 * config.n_layer)
                nn.init.normal_(param, mean=0.0, std=std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        idx: torch.Tensor,            # (B, T) token indices
        targets: torch.Tensor = None, # (B, T) shifted targets for LM loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} > block_size {self.config.block_size}"
        )
        device = idx.device
        pos = torch.arange(T, device=device)

        # ── Forward pass ──────────────────────────────────────────────────────
        tok_emb = self.transformer["wte"](idx)          # (B, T, n_embd)
        pos_emb = self.transformer["wpe"](pos)          # (T, n_embd) → broadcast
        x = self.transformer["drop"](tok_emb + pos_emb)

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Training — compute cross-entropy over all positions
            logits = self.lm_head(x)                    # (B, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference — only decode the last token (efficiency)
            logits = self.lm_head(x[:, [-1], :])       # (B, 1, vocab)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Auto-regressive generation with temperature + top-k + nucleus sampling.
        
        Args:
            idx:            (B, T) seed token indices
            max_new_tokens: number of tokens to generate
            temperature:    >1 = more random, <1 = more deterministic
            top_k:          keep only k highest logit tokens
            top_p:          nucleus — keep tokens covering top p probability mass
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # (B, vocab)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                # Remove tokens beyond nucleus
                sorted_idx_to_remove = cum_probs - probs > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        p = self.count_parameters()
        return (
            f"GPT({self.config!r})\n"
            f"  Total params:     {p['total']/1e6:.2f}M\n"
            f"  Trainable params: {p['trainable']/1e6:.2f}M"
        )
