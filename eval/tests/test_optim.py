import pytest
import torch
from src.vanilla.config import GPTConfig
from src.vanilla.model import GPT
from training.optim_utils import clip_gradient_norm, get_weight_decay_params

def test_gradient_clipping_logic():
    cfg = GPTConfig.micro()
    model = GPT(cfg)
    
    # Fake gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 10.0  # Huge gradients
    
    norm = clip_gradient_norm(model, max_norm=1.0)
    assert norm > 1.0  # Before clip it should be huge
    
    # Assert clipping took place (checking one parameter)
    for p in model.parameters():
        assert p.grad.abs().max().item() < 10.0
        break

def test_weight_decay_grouping():
    cfg = GPTConfig.micro()
    model = GPT(cfg)
    optim_groups = get_weight_decay_params(model, weight_decay=0.1)
    
    assert len(optim_groups) == 2
    decay_params = optim_groups[0]["params"]
    no_decay_params = optim_groups[1]["params"]
    
    # Ensure mutually exclusive and collectively exhaustive
    assert len(decay_params) > 0
    assert len(no_decay_params) > 0
    assert len(decay_params) + len(no_decay_params) == len(list(model.parameters()))
