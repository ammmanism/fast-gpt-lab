"""
VRAM Auditing & Fragmentation — fast-gpt-lab
Tracks PyTorch memory allocator to measure fragmentation and peak VRAM.
"""
import torch
import gc

class MemoryAuditor:
    """
    Diagnoses OutOfMemory (OOM) errors and measures fragmentation ratio.
    A high fragmentation ratio means memory is physically available but scattered
    into unusable tiny chunks.
    """
    @staticmethod
    def reset_peaks():
        """Reset maximum memory trackers at the start of a profiling epoch."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    @staticmethod
    def get_vram_status(device: int = 0) -> dict:
        """Returns peak memory, currently allocated memory, and fragmentation ratio."""
        if not torch.cuda.is_available():
            return {"error": "CUDA unavailable"}

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        
        # Fragmentation = Reserved memory that the OS gave to PyTorch, 
        # but PyTorch isn't currently using for active tensors.
        if reserved > 0:
            fragmentation_ratio = (reserved - allocated) / reserved
        else:
            fragmentation_ratio = 0.0

        return {
            "allocated_mb": allocated / (1024 ** 2),
            "reserved_mb": reserved / (1024 ** 2),
            "peak_allocated_mb": peak_allocated / (1024 ** 2),
            "fragmentation_percent": fragmentation_ratio * 100.0
        }

    @staticmethod
    def force_garbage_collection():
        """Hard refresh. Destroys cached graphs and forces Python/C++ to release pointers."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
