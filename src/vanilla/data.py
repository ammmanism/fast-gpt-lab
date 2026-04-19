"""
Data Loading Utilities — fast-gpt-lab
Memory-mapped binary token files for zero-copy, high-throughput data loading.
"""
import os
import numpy as np
import torch
from pathlib import Path


class DataLoader:
    """
    Infinite iterator over memory-mapped token shards.
    
    Token files are pre-tokenized NumPy uint16 arrays stored as raw binary.
    Memory mapping means the OS handles paging — training process stays lean.
    
    Usage:
        loader = DataLoader("train", "data/", batch_size=12, block_size=1024, device="cuda")
        x, y = next(loader)   # shape: (B, T), (B, T)
    """

    def __init__(
        self,
        split: str,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: str | torch.device,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # Find all shard files for this split
        data_dir = Path(data_dir)
        shards = sorted(data_dir.glob(f"{split}_*.bin"))
        assert shards, f"No data shards found for split '{split}' in {data_dir}"
        self.shards = shards

        self._shard_idx = 0
        self._load_shard(self.shards[self._shard_idx])
        self._ptr = 0
        print(f"📦 DataLoader [{split}]: {len(shards)} shards, {len(self.tokens):,} tokens")

    def _load_shard(self, path: Path) -> None:
        self.tokens = np.memmap(str(path), dtype=np.uint16, mode="r")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.block_size
        needed = B * T + 1

        # Advance shard if needed
        if self._ptr + needed > len(self.tokens):
            self._shard_idx = (self._shard_idx + 1) % len(self.shards)
            self._load_shard(self.shards[self._shard_idx])
            self._ptr = 0

        buf = self.tokens[self._ptr : self._ptr + needed].astype(np.int64)
        self._ptr += B * T

        x = torch.from_numpy(buf[:-1].reshape(B, T)).to(self.device)
        y = torch.from_numpy(buf[1:].reshape(B, T)).to(self.device)
        return x, y


def prepare_tinystories(data_dir: str = "data/") -> None:
    """Download and tokenize TinyStories for quick iteration."""
    import urllib.request, gzip, shutil

    os.makedirs(data_dir, exist_ok=True)
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    archive = os.path.join(data_dir, "tinystories.tar.gz")
    if not os.path.exists(archive):
        print("⬇️  Downloading TinyStories...")
        urllib.request.urlretrieve(url, archive)
    print("✅ TinyStories ready. Run tokenize_dataset.py to prepare binary shards.")


def prepare_openwebtext(data_dir: str = "data/", num_proc: int = 8) -> None:
    """Download and tokenize OpenWebText (full GPT-2 pretraining dataset)."""
    try:
        from datasets import load_dataset
        import tiktoken
    except ImportError:
        raise RuntimeError("pip install datasets tiktoken")

    enc = tiktoken.get_encoding("gpt2")

    print("⬇️  Fetching OpenWebText (~55GB)...")
    ds = load_dataset("openwebtext", num_proc=num_proc)

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # end-of-text separator
        return {"ids": ids, "len": len(ids)}

    ds = ds.map(tokenize, remove_columns=["text"], num_proc=num_proc, desc="Tokenizing")

    split_ds = ds["train"].train_test_split(test_size=0.0005, seed=2357)
    split_ds["val"] = split_ds.pop("test")

    os.makedirs(data_dir, exist_ok=True)
    for split, dataset in split_ds.items():
        total_tokens = sum(dataset["len"])
        out = np.memmap(
            os.path.join(data_dir, f"{split}_00000.bin"),
            dtype=np.uint16, mode="w+", shape=(total_tokens,)
        )
        idx = 0
        for batch in dataset.iter(batch_size=1024):
            for ids in batch["ids"]:
                out[idx : idx + len(ids)] = ids
                idx += len(ids)
        out.flush()
        print(f"  ✅ {split}: {total_tokens:,} tokens → {split}_00000.bin")
