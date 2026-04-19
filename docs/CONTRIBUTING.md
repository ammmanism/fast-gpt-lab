# Contributing to fast-gpt-lab

We welcome contributions that improve **speed, accuracy, or pedagogical clarity**.
This is a high-standards codebase — please read the full guide before opening a PR.

---

## Development Setup

```bash
git clone https://github.com/your-username/fast-gpt-lab
cd fast-gpt-lab
# Install uv if not present: https://github.com/astral-sh/uv
uv sync
pre-commit install
```

## Branch Strategy

```
main        ← stable, tagged releases only
develop     ← integration branch (all PRs target here)
feat/*      ← new features (e.g. feat/rope-embeddings)
fix/*       ← bug fixes
bench/*     ← benchmark improvements
docs/*      ← documentation only
```

## Code Standards

### Mandatory
- **Type hints** on every public function
- **Docstring** explaining *what* and *why* (not just *what*)
- **Inline math comments** for any numerical operation
- Tests in `eval/` for new evaluation functions
- `torch.allclose` validation for new Triton kernels

### Commit Message Format (Conventional Commits)
```
type(scope): short description

Longer explanation if needed.
References: paper/issue
```
Types: `feat`, `fix`, `perf`, `docs`, `bench`, `refactor`, `chore`, `test`

### Examples
```
feat(kernels): add Triton RoPE kernel with fused sin/cos
perf(attention): reduce SDPA overhead by fusing QKV projection
docs(theory): add BPE algorithm complexity proof
bench: compare RoPE vs ALiBi on 8k context
```

## Performance Bar

Every performance-critical PR must include:
1. `make bench` output before and after
2. MFU% delta (must not regress by > 2%)
3. Memory usage table

## Running Tests

```bash
make test          # full test suite
make check         # lint + type check
make bench         # throughput benchmark
make profile       # generate Chrome trace
```

## Submitting a PR

1. Branch from `develop`
2. Fill out the PR template completely
3. Ensure all CI checks pass
4. Request review from `@codeowners`

---

Thank you for making fast-gpt-lab faster. ⚡
