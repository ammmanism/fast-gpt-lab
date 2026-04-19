# --- Makefile for fast-gpt-lab ---

.PHONY: setup train bench profile test check clean

# Virtual Environment & Dependencies
setup:
	@echo "🚀 Setting up environment with uv..."
	uv venv
	uv sync

# Training Entrypoint
train:
	@echo "🔥 Starting distributed training..."
	uv run python -m src.training.train --config configs/base.yaml

# Performance Benchmarking
bench:
	@echo "⚡ Running speed benchmarks..."
	uv run python benchmarks/run_bench.py --compare nanogpt

# MFU & Memory Profiling
profile:
	@echo "📊 Profiling MFU and Memory usage..."
	uv run python profiling/audit.py --output profiling/latest_trace.json

# Quality Checks
check:
	@echo "🔍 Linting and Type checking..."
	uv run ruff check .
	uv run mypy src

# Tests
test:
	@echo "🧪 Running scientific validation..."
	uv run pytest eval/

# Cleanup
clean:
	@echo "🧹 Cleaning up artifacts..."
	rm -rf `find . -name __pycache__`
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
