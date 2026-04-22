# ADR-015: Gateway API & Token Streaming (SSE)

**Status**: Accepted  
**Date**: 2026-04-22  
**Deciders**: Core Team

---

## Context

Running the model internally in Python via `model.generate()` is adequate for benchmarking, but insufficient for user-facing applications. We need a standardized way for web clients, TUIs, and external services to interact with our LLM with minimal perceived latency.

## Decision

We adopt **FastAPI** as our deployment gateway. To mitigate the "time to first token" (TTFT) latency, we implement Server-Sent Events (SSE) via FastAPI's `StreamingResponse`. This allows chunks (tokens) to be pushed to the client immediately as they are generated from the `model()` forward pass loop, rather than waiting for the entire sequence to finish generating.

## Consequences
- ✅ TTFT is drastically reduced to the sub-100ms range.
- ✅ Allows seamless integration with modern chat frontends (React/Next.js) & TUIs.
- ❌ Naive generation loops block the async event loop; requires background thread offloading or vLLM-style continuous batching for true high-concurrency production scales.
