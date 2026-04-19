"""
Perplexity Evaluation — fast-gpt-lab
Evaluates language model perplexity on WikiText-103 and Penn Treebank.

Perplexity: PPL = exp(-1/N Σ log P(w_t | w_<t))
Lower is better. GPT-2-117M achieves ~29.4 on WikiText-103.
"""
import math
import torch
from pathlib import Path


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataset: str = "wikitext-103",
    split: str = "test",
    stride: int = 512,
    max_tokens: int = 2_097_152,  # 2M tokens cap
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """
    Sliding-window perplexity (avoids boundary effects).
    
    Uses stride < block_size to get full-context predictions for every token.
    This is the standard evaluation protocol used in GPT-2 and LLaMA papers.
    
    Args:
        stride: number of new tokens predicted per window (< block_size)
    Returns:
        Perplexity score (float)
    """
    model.eval()
    block_size = model.config.block_size

    # Load tokens
    tokens = _load_tokens(dataset, split)
    tokens = tokens[:max_tokens]
    N = len(tokens)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)

    nlls = []
    prev_end = 0

    for begin in range(0, N, stride):
        end   = min(begin + block_size, N)
        seq   = input_ids[begin:end]
        trg   = input_ids[begin + 1 : end + 1]
        if len(seq) < 2:
            break

        # Only count tokens predicted with full context
        target_len = end - max(prev_end, begin + 1)
        prev_end = end

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(seq.unsqueeze(0))

        # Slice logits for new tokens only
        shift_logits = logits[0, -target_len - 1 : -1, :]
        shift_labels = trg[-target_len:]
        if shift_labels.numel() == 0:
            continue

        nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="sum")
        nlls.append(nll.item())

    total_nll = sum(nlls)
    ppl = math.exp(total_nll / N)
    print(f"📊 Perplexity on {dataset}/{split}: {ppl:.2f}")
    return ppl


def _load_tokens(dataset: str, split: str) -> list[int]:
    """Load pre-tokenized token file or download on first use."""
    cache_path = Path(f".cache/eval/{dataset.replace('/', '_')}_{split}.bin")
    if cache_path.exists():
        import numpy as np
        return np.fromfile(str(cache_path), dtype=np.int32).tolist()

    if dataset in ("wikitext-103", "wikitext-2"):
        return _download_wikitext(dataset, split, cache_path)
    raise ValueError(f"Unknown dataset: {dataset}")


def _download_wikitext(dataset: str, split: str, cache_path: Path) -> list[int]:
    from datasets import load_dataset
    import tiktoken
    import numpy as np

    print(f"⬇️  Downloading {dataset}...")
    ds_name = "wikitext" if "wikitext" in dataset else dataset
    config  = "wikitext-103-raw-v1" if "103" in dataset else "wikitext-2-raw-v1"
    ds = load_dataset(ds_name, config, split=split)

    enc = tiktoken.get_encoding("gpt2")
    all_ids = []
    for row in ds:
        text = row["text"].strip()
        if text:
            all_ids.extend(enc.encode_ordinary(text))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.array(all_ids, dtype=np.int32).tofile(str(cache_path))
    print(f"  ✅ Cached {len(all_ids):,} tokens → {cache_path}")
    return all_ids
