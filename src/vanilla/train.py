"""
Vanilla Training Loop — fast-gpt-lab
Clean, readable training script with gradient accumulation, AMP, and W&B logging.
"""
import os
import math
import time
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import GPTConfig
from .model import GPT
from .data import DataLoader


# ─── Training Config ─────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    dataset: str = "openwebtext"
    data_dir: str = "data/"
    block_size: int = 1024

    # Batch
    batch_size: int = 12
    grad_accum_steps: int = 40      # effective batch = 12 * 40 = ~500k tokens

    # Optimiser
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 600000
    lr_decay_steps: int = 600000
    min_lr: float = 6e-5

    # Evaluation
    eval_interval: int = 2000
    eval_steps: int = 200
    log_interval: int = 10

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"         # bfloat16 | float16 | float32
    compile: bool = True             # torch.compile for ~30% throughput gain

    # Checkpointing
    out_dir: str = "checkpoints/"
    checkpoint_interval: int = 5000

    # Logging
    wandb: bool = False
    wandb_project: str = "fast-gpt-lab"
    wandb_run_name: str = "gpt2-124M"


# ─── Learning Rate Schedule ───────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Cosine decay with linear warmup.
    
    Schedule:
        [0, warmup_steps)      → linear ramp 0 → lr
        [warmup_steps, decay)  → cosine decay lr → min_lr
        [decay, ∞)             → constant min_lr
    """
    # Linear warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps

    # Constant floor after decay
    if step > cfg.lr_decay_steps:
        return cfg.min_lr

    # Cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─── Optimiser (decoupled weight decay) ─────────────────────────────────────

def configure_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """
    AdamW with decoupled weight decay.
    
    Weight decay is applied ONLY to 2D params (weight matrices).
    Excluded: biases, LayerNorm {weight, bias}, embeddings.
    
    Rationale: Embedding tables & norm params are not "overfit" candidates.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=use_fused,   # CUDA fused kernel — ~25% faster on A100
    )


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(model_cfg: GPTConfig, train_cfg: TrainConfig) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    device = torch.device(train_cfg.device)
    torch.manual_seed(1337)

    # dtype context for AMP
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[train_cfg.dtype]
    ctx = torch.autocast(device_type=train_cfg.device, dtype=ptdtype) if train_cfg.device == "cuda" else nullcontext()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = DataLoader("train", train_cfg.data_dir, train_cfg.batch_size, model_cfg.block_size, device)
    val_loader   = DataLoader("val",   train_cfg.data_dir, train_cfg.batch_size, model_cfg.block_size, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPT(model_cfg).to(device)
    if train_cfg.compile:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model)  # noqa

    optimizer = configure_optimizer(model, train_cfg)

    # ── Optional W&B ──────────────────────────────────────────────────────────
    if train_cfg.wandb:
        import wandb
        wandb.init(project=train_cfg.wandb_project, name=train_cfg.wandb_run_name)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t_start = time.perf_counter()
    tokens_seen = 0
    best_val_loss = float("inf")

    for step in range(train_cfg.max_steps + 1):
        # Dynamic LR
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % train_cfg.eval_interval == 0:
            losses = _evaluate(model, val_loader, train_cfg, ctx)
            val_loss = losses["val"]
            print(f"[step {step:6d}] train={losses['train']:.4f} val={val_loss:.4f} lr={lr:.2e}")
            if train_cfg.wandb:
                import wandb
                wandb.log({"train/loss": losses["train"], "val/loss": val_loss, "lr": lr, "step": step})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, step, val_loss, train_cfg)

        # ── Gradient accumulation ─────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(train_cfg.grad_accum_steps):
            x, y = next(train_loader)
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()

        # Gradient clipping (prevents exploding gradients)
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()
        tokens_seen += train_cfg.batch_size * model_cfg.block_size * train_cfg.grad_accum_steps

        if step % train_cfg.log_interval == 0:
            t_now = time.perf_counter()
            tok_per_sec = tokens_seen / (t_now - t_start)
            print(f"  step {step:6d} | loss {loss_accum.item():.4f} | {tok_per_sec/1e3:.1f}k tok/s")


@torch.no_grad()
def _evaluate(model, loader, cfg: TrainConfig, ctx) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, dl in [("train", loader), ("val", loader)]:
        total_loss = 0.0
        for _ in range(cfg.eval_steps):
            x, y = next(dl)
            with ctx:
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / cfg.eval_steps
    model.train()
    return losses


def _save_checkpoint(model, optimizer, step, val_loss, cfg: TrainConfig) -> None:
    path = os.path.join(cfg.out_dir, f"ckpt_step{step:06d}_loss{val_loss:.4f}.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"  💾 Saved checkpoint → {path}")
