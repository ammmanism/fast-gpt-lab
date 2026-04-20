# ADR-009: Streaming Datasets (Iterable Loading)

**Status**: Accepted  
**Date**: 2026-04-20  
**Deciders**: Core Team

---

## Context

Training a foundational model requires massive corpora. For example, RedPajama is several terabytes large. Downloading and locally caching these datasets creates I/O bottlenecks and requires exorbitant nvme storage allocations on every single worker node in a distributed cluster.

## Decision

We introduce `StreamingDataLoader` via HuggingFace's `datasets` library with `streaming=True`. Instead of materializing data locally, we yield tokens directly from the network stream, buffer them to construct blocks, and dispatch them to the GPU.

## Consequences
- ✅ Eliminates local storage requirements for datasets.
- ✅ Instantaneous training start times (no waiting for 5-hour dataset preparations).
- ❌ Introduces network bandwidth dependency. If the cluster lacks high-speed internet, training steps will stall on I/O.
- ❌ We cannot easily shuffle the *entire* dataset globally; only local buffer shuffling is possible.
