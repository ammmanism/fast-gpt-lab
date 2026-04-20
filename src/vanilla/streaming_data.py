"""
Streaming Data Loader — fast-gpt-lab
Streams terabyte-scale datasets directly from HuggingFace without local caching.
"""
import torch
from typing import Iterator, Dict, Any

class StreamingDataLoader:
    """
    Wraps a HuggingFace iterable dataset for infinite streaming.
    Bypasses the need for large local storage when training on massive corpora like RedPajama.
    """
    def __init__(self, dataset_name: str, batch_size: int, block_size: int, device: str = "cuda"):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.iterator = self._build_iterator()
        
    def _build_iterator(self):
        try:
            from datasets import load_dataset
            import tiktoken
        except ImportError:
            raise RuntimeError("Requires `pip install datasets tiktoken`")
            
        enc = tiktoken.get_encoding("gpt2")
        dataset = load_dataset(self.dataset_name, streaming=True, split="train")
        
        tokenized_buffer = []
        
        for sample in dataset:
            text = sample.get("text", "")
            if not text:
                continue
                
            tokens = enc.encode_ordinary(text)
            tokens.append(enc.eot_token)
            tokenized_buffer.extend(tokens)
            
            # Yield when buffer has enough for a batch
            chunk_size = self.batch_size * (self.block_size + 1)
            while len(tokenized_buffer) >= chunk_size:
                chunk = tokenized_buffer[:chunk_size]
                tokenized_buffer = tokenized_buffer[chunk_size:]
                
                # Shape: (B, T+1)
                t_tensor = torch.tensor(chunk, dtype=torch.long).view(self.batch_size, self.block_size + 1)
                
                x = t_tensor[:, :-1].to(self.device, non_blocking=True)
                y = t_tensor[:, 1:].to(self.device, non_blocking=True)
                yield x, y

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            # Rebuild dataset iterator on epoch boundary
            self.iterator = self._build_iterator()
            return next(self.iterator)
