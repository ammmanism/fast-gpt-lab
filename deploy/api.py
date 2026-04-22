"""
FastAPI High-Performance Gateway — fast-gpt-lab
Implements Server-Sent Events (SSE) for token streaming and continuous batching scaffolding.
"""
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import time

from src.vanilla.config import GPTConfig
from src.vanilla.model import GPT
from src.tokenizer.bpe import BPETokenizer

app = FastAPI(title="FastGPT-Lab Inference Gateway", version="1.0.0")

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95

# Global model state (in a real production app, this would be managed by a dedicated worker pool)
MODEL_STATE = {}

@app.on_event("startup")
async def load_model():
    print("🚀 Initializing Inference Engine...")
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    tokenizer = BPETokenizer()
    MODEL_STATE["model"] = model
    MODEL_STATE["tokenizer"] = tokenizer
    MODEL_STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Model loaded and ready.")

async def token_stream_generator(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    """Yields tokens as they are generated using Server-Sent Events (SSE)."""
    model = MODEL_STATE["model"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Autoregressive generation loop
    # Note: In a production cluster, vLLM continuous batching replaces this naive loop
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            
        # Focus on the last time step
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply Top-P sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to mask
        mask = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        next_token_logits[mask] = -float('Inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        
        # Decode the single token
        decoded_word = tokenizer.decode([next_token.item()])
        
        yield f"data: {decoded_word}\n\n"
        
        # Artificial delay to prevent event loop starvation during naive generation
        await asyncio.sleep(0.01)

@app.post("/v1/completions/stream")
async def stream_completions(req: GenerationRequest):
    """SSE Endpoint for chat-like token streaming."""
    if not MODEL_STATE.get("model"):
        raise HTTPException(status_code=503, detail="Model is currently loading.")
        
    return StreamingResponse(
        token_stream_generator(req.prompt, req.max_new_tokens, req.temperature, req.top_p),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "FastGPT-Lab Kernel"}
