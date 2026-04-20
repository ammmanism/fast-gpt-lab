# ADR-008: Enterprise Telemetry integration (Weights & Biases)

**Status**: Accepted  
**Date**: 2026-04-20  
**Deciders**: Core Team

---

## Context

Training runs on clusters can fail silently, hardware can degrade, and loss curves must be monitored remotely. We need a system to record granular metrics (Loss, MFU, Gradient Norms, Learning Rate) and sync them to a remote dashboard in real-time, while remaining fault-tolerant if offline.

## Decision

We adopt **Weights & Biases (wandb)** as our primary telemetry layer (`training/telemetry.py`). We implement a wrapper class to guarantee that if the W&B dependency is missing, or the cluster node loses internet connection, the training run continues without crashing (graceful fallback).

## Consequences
- ✅ Beautiful visualizations and artifact tracking for production readiness.
- ✅ Granular gradient norm tracking to diagnose explosion/vanishing gradients.
- ❌ Introduces a 3rd party dependency for full tracking capability.
