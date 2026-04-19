"""
DDP (Distributed Data Parallel) training utilities — fast-gpt-lab
Simpler than FSDP: all ranks hold full model copy, sync gradients only.
Best for models < 1B params where model fits in GPU VRAM.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_setup() -> tuple[int, int]:
    """Setup NCCL backend. Returns (rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"🔥 DDP: {world_size} GPUs")
    return rank, world_size


def wrap_ddp(model: torch.nn.Module, rank: int) -> DDP:
    model = model.to(rank)
    return DDP(model, device_ids=[rank], find_unused_parameters=False)


def ddp_teardown() -> None:
    dist.destroy_process_group()


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Average loss across all DDP ranks."""
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    return loss
