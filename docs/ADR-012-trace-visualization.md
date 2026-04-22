# ADR-012: Trace Visualization for Kernel Debugging

**Status**: Accepted  
**Date**: 2026-04-21  
**Deciders**: Core Team

---

## Context

Metrics like "Time per step" are too coarse and obscure the root cause of latency. If a step takes 250ms, we don't know if the bottleneck is dataloader overhead, GPU synchronization, NCCL ring delays, or naive grid execution on matrix multiplications. We need visual, chronological telemetry.

## Decision

We use `torch.profiler` configured with `torch.profiler.tensorboard_trace_handler` to generate Chrome JSON Trace files. These traces will be parsed by `chrome_trace_analyzer.py` internally for regression testing, but more importantly, manually inspected using `chrome://tracing` or Perfetto.

## Consequences
- ✅ Provides nanosecond-level visibility into exactly which GPU Streaming Multiprocessors (SMs) are active.
- ✅ Exposes overhead from CPU-to-GPU memory copies explicitly.
- ❌ Trace files are massive (often >500MB per step), requiring isolation to only the 10th training step (warmup completed).
