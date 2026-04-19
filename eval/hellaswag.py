"""
HellaSwag Evaluation — fast-gpt-lab
Zero-shot evaluation of commonsense reasoning (Zellers et al., 2019).

Task: given a partial sentence + 4 continuations, pick the most likely one.
GPT-2-117M: 29.6% | GPT-2-345M: 40.9% | GPT-3-175B: 79.3%
"""
import json
import torch
import tiktoken
from pathlib import Path


@torch.no_grad()
def evaluate_hellaswag(
    model: torch.nn.Module,
    data_path: str = "data/hellaswag_val.jsonl",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_samples: int = None,
) -> float:
    """
    Zero-shot HellaSwag evaluation via log-likelihood scoring.
    
    Scoring: argmax_i P(continuation_i | context)
    
    We compute the normalized log-probability of each continuation given the
    context (normalization by token count for length-fairness).
    
    Returns: accuracy (0..1)
    """
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    data = _load_hellaswag(data_path)
    if num_samples:
        data = data[:num_samples]

    correct = 0
    for i, item in enumerate(data):
        context = item["ctx"]
        endings = item["endings"]
        label   = int(item["label"])

        ctx_ids = enc.encode(context)
        scores  = []

        for ending in endings:
            end_ids = enc.encode(" " + ending)  # leading space = GPT tokenizer convention
            full_ids = torch.tensor(ctx_ids + end_ids, dtype=torch.long, device=device)

            # Compute token-by-token log-probability
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(full_ids[:-1].unsqueeze(0))

            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            # Only score the completion tokens
            ending_log_probs = log_probs[len(ctx_ids) - 1:, :]
            ending_token_ids = full_ids[len(ctx_ids):]
            if ending_token_ids.numel() == 0:
                scores.append(float("-inf"))
                continue
            score = ending_log_probs[range(len(ending_token_ids)), ending_token_ids].mean().item()
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(data)}] accuracy: {acc:.3f}")

    final_acc = correct / len(data)
    print(f"\n📊 HellaSwag accuracy: {final_acc*100:.2f}%")
    return final_acc


def _load_hellaswag(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        _download_hellaswag(p)
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _download_hellaswag(dest: Path) -> None:
    import urllib.request
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading HellaSwag validation set...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  ✅ Saved to {dest}")
