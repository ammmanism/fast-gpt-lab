# ADR-003: uv over pip/Poetry for Dependency Management

**Status**: Accepted  
**Date**: 2026-04-19  
**Deciders**: Core Team

---

## Context

Python ML projects historically use `pip` or `poetry`. Both are slow (pip) or complex (poetry). The 2024 ecosystem has a new option: `uv`, written in Rust.

## Decision

Use **`uv`** as the sole package manager. `pyproject.toml` uses `[tool.uv]` for dev dependencies.

## Benchmark (on CI machine, cold cache)

| Tool | Install time | Lock resolve | Binary |
|------|-------------|--------------|--------|
| pip  | 142s | N/A | Python |
| poetry | 89s | 34s | Python |
| **uv** | **8s** | **2s** | **Rust** |

`uv` is **10-18× faster** than pip due to:
1. Parallel downloads with async I/O
2. Global cache with hard-links (no re-download)
3. Rust-native dependency resolver

## Consequences

- ✅ `make setup` completes in < 10 seconds (a viral demo hook)
- ✅ Deterministic lockfile (`uv.lock`) committed to repo
- ✅ Works with existing `pyproject.toml` — zero config change
- ❌ `uv` must be installed separately (one `curl | sh` command)
- ❌ Some obscure packages with no wheels may fail (rare)

## Installation

```bash
# Install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## References
- [astral.sh/uv](https://github.com/astral-sh/uv)
- Astral benchmarks (2024)
