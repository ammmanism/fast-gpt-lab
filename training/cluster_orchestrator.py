"""
Cluster Orchestrator — fast-gpt-lab
SLURM and MPI environment variables bootstrap for multi-node training.
"""
import os
import torch
import torch.distributed as dist

class ClusterOrchestrator:
    """
    Automatically detects SLURM or MPI variables to initialize torch.distributed
    without needing manual `torchrun` wrapper over multiple physical nodes.
    """
    @staticmethod
    def initialize():
        if "SLURM_PROCID" in os.environ:
            return ClusterOrchestrator._init_slurm()
        elif "OMPI_COMM_WORLD_RANK" in os.environ:
            return ClusterOrchestrator._init_mpi()
        else:
            return ClusterOrchestrator._init_local()

    @staticmethod
    def _init_slurm():
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % torch.cuda.device_count()))
        
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Determine master address
        hostnames = os.popen("scontrol show hostnames $SLURM_JOB_NODELIST").read().split()
        if hostnames:
            os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    @staticmethod
    def _init_mpi():
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    @staticmethod
    def _init_local():
        # Fallback to local torchrun or single process
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
