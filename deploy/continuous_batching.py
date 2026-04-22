"""
Continuous Batching Orchestrator — fast-gpt-lab
vLLM-style request pooling for maximum high-throughput API serving.
"""
import asyncio
from typing import List, Dict

class RequestContext:
    def __init__(self, request_id: str, prompt_tokens: List[int]):
        self.request_id = request_id
        self.tokens = prompt_tokens
        self.generated = []
        self.is_finished = False

class ContinuousBatchEngine:
    """
    Unlike naive inference where a batch must finish completely before the next one starts,
    Continuous Batching (Orca/vLLM) dynamically inserts new requests into the GPU queue
    during the middle of a generation cycle. This prevents Head-of-Line blocking.
    """
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.waiting_queue = asyncio.Queue()
        self.active_batch: List[RequestContext] = []
        
    async def add_request(self, request_id: str, tokens: List[int]):
        """Ingests new network traffic requests instantly."""
        ctx = RequestContext(request_id, tokens)
        await self.waiting_queue.put(ctx)
        print(f"📥 Queued Request: {request_id}")

    async def _step_generation(self):
        """Simulates the GPU step processing the active batch."""
        # 1. Pull waiting requests into active batch if space permits
        while len(self.active_batch) < self.max_batch_size and not self.waiting_queue.empty():
            new_req = await self.waiting_queue.get()
            self.active_batch.append(new_req)
            
        if not self.active_batch:
            return

        # 2. Simulate parallel GPU generation (Matrix Multiplication across all contexts)
        # In actual deployment, this invokes the model(input_ids) loop.
        await asyncio.sleep(0.05) 
        
        # 3. Prune finished requests to free slots dynamically
        retained = []
        for req in self.active_batch:
            req.generated.append(1) # Fake token
            if len(req.generated) >= 10: # arbitrary termination
                req.is_finished = True
                print(f"✅ Finished: {req.request_id}")
            else:
                retained.append(req)
                
        self.active_batch = retained

    async def start_engine(self):
        """Infinite background loop attached to the API worker."""
        print("⚙️ Continuous Batching Engine Online.")
        while True:
            await self._step_generation()
            await asyncio.sleep(0.01)
