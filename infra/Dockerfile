FROM python:3.11-slim

LABEL maintainer="fast-gpt-lab <lab@fast-gpt.ai>"
LABEL description="fast-gpt-lab: Elite GPT training environment"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Install dependencies (cached layer) ──────────────────────────────────────
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["uv", "run", "python"]
CMD ["-m", "src.vanilla.train"]
