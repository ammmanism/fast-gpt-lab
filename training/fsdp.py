"""
FSDP Distributed Training — fast-gpt-lab
Fully Sharded Data Parallel for training models across multiple GPUs.

FSDP shards parameters, gradients, and optimizer states across all ranks.
Memory per GPU: O(N/world_size) instead of O(N) — enables 7B+ model training.
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import functools


def setup_distributed() -> tuple[int, int]:
    """Initialize NCCL distributed process group. Returns (rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"🔥 Distributed setup: {world_size} GPUs via NCCL")
    return rank, world_size


def teardown_distributed() -> None:
    dist.destroy_process_group()


def wrap_model_fsdp(
    model: torch.nn.Module,
    rank: int,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
) -> FSDP:
    """
    Wrap a model with FSDP.
    
    Sharding strategies:
        FULL_SHARD: shard params + grads + optimizer → max memory savings
        SHARD_GRAD_OP: shard grads + optimizer only → less comm overhead
        NO_SHARD: DDP-equivalent, no memory savings
        
    Args:
        model:            nn.Module to wrap
        rank:             current process rank
        sharding_strategy: "FULL_SHARD" | "SHARD_GRAD_OP" | "NO_SHARD"
        cpu_offload:      offload optimizer state to CPU (slow but saves VRAM)
        mixed_precision:  use bfloat16 for forward/backward
    """
    strategy_map = {
        "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD":      ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map[sharding_strategy]

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Auto-wrap transformer blocks at size threshold
    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000)

    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap comm with compute
        mixed_precision=mp_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=rank,
        use_orig_params=True,   # required for torch.compile compatibility
    )

    if rank == 0:
        print(f"🛡️  FSDP wrapped — strategy={sharding_strategy}, bf16={mixed_precision}")

    return fsdp_model


def save_fsdp_checkpoint(model: FSDP, optimizer, step: int, path: str, rank: int) -> None:
    """
    Save full (un-sharded) checkpoint.
    Requires FSDP's full state dict consolidation.
    Only rank 0 writes to disk.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"step": step, "model": state, "optimizer": optimizer.state_dict()}, path)
        print(f"  💾 FSDP checkpoint saved at step {step} → {path}")

    dist.barrier()  # Ensure all ranks sync before continuing


def load_fsdp_checkpoint(model: FSDP, path: str, rank: int) -> int:
    """Load checkpoint into FSDP model. Returns the saved step."""
    ckpt = torch.load(path, map_location="cpu")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(ckpt["model"])
    if rank == 0:
        print(f"  ✅ Loaded checkpoint from step {ckpt['step']} → {path}")
    return ckpt["step"]
