"""
Optimization Utilities — fast-gpt-lab
Gradient clipping, norm regularisation, and specialized schedulers.
"""
import torch

def clip_gradient_norm(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    Returns the total scalar norm of the parameters' gradients before clipping.
     Essential for preventing exploding gradients in deep transformers.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]),
        2.0
    ).item()

    # Clip in-place
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm

def get_weight_decay_params(model: torch.nn.Module, weight_decay: float = 0.1):
    """
    Decoupled weight decay specification.
    We apply weight decay ONLY to matrix multiplications (2D+ tensors).
    Biases and LayerNorm scales shouldn't decay, as it reduces capacity.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    # Some modules might not be captured perfectly. Fallback check:
    param_dict = {pn: p for pn, p in model.named_parameters()}
    for pn, p in param_dict.items():
        if pn not in decay and pn not in no_decay:
            # Squeeze and checks implicitly assign
            if p.dim() < 2:
                no_decay.add(pn)
            else:
                decay.add(pn)

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    return optim_groups
