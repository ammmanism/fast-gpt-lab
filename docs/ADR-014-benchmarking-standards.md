# ADR-014: Official Benchmarking Pipeline & Trace Serialization

**Status**: Accepted  
**Date**: 2026-04-21  
**Deciders**: Core Team

---

## Context

We cannot rely on ad-hoc claims of "It runs fast." We need reproducible evidence that compares PyTorch conventional implementations strictly against our custom Triton kernels, logged out directly via Chrome JSON trace visualization grids.

## Decision

We introduce `benchmarks/run_profiler.py` utilizing the `libkineto` backend of PyTorch (`torch.profiler`). All traces will be exported into the `./profiling_logs` directory via a shared `TraceExporter` singleton to avoid local storage exhaustion (traces capture millions of microsecond CUDA events and easily blow up disk space if not scheduled cleanly using the warmup/active sliding window).

## Consequences
- ✅ Deeply objective, highly shareable visual artifacts.
- ✅ Explicit Multiplicative Speedup factor definitions in `compare_throughput.py`.
- ❌ Trace serialization incurs a 400% runtime overhead *during* profiling steps. Profiler must remain utterly severed from standard production `train.py` loop execution.
