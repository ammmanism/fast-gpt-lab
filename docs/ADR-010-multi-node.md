# ADR-010: Multi-Node Orchestration

**Status**: Accepted  
**Date**: 2026-04-20  
**Deciders**: Core Team

---

## Context

Scaling highly parameter-intensive transformer models forces training pipelines to stretch across geographic physical boundaries (multiple server racks). We need our training loop to be capable of seamless multi-node deployment natively, without heavily relying exclusively on `torchrun`.

## Decision

We decouple internal distributed initialization from the PyTorch launch utility by writing our own `ClusterOrchestrator`. It passively detects distributed environment variables injected dynamically by HPC schedulers like SLURM (`SLURM_PROCID`) and OpenMPI (`OMPI_COMM_WORLD_RANK`), extracting them to boot the `nccl` backend cleanly.

## Consequences
- ✅ Agnostic scaling — engineers can train on physical bare metal, K8s, or traditional supercomputer SLURM queues.
- ✅ Simplifies Docker deployment entrypoints by bypassing the `torchrun` wrapper script.
- ❌ Forces us to manually build multi-node fallback routines if environmental injection is malformed.
