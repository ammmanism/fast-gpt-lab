## 🔀 PR Summary
<!-- One line: what does this PR do and why? -->

## 📊 Performance Impact
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Tokens/sec | | | |
| MFU (%) | | | |
| Memory (GB) | | | |
| Perplexity | | | |

## 🧪 Testing
- [ ] Unit tests pass (`make test`)
- [ ] Linting clean (`make check`)
- [ ] Benchmark regression within ±2% (`make bench`)
- [ ] Added/updated docstring for any new function
- [ ] Theoretical justification documented if adding new optimization

## 📐 Mathematical Correctness
<!-- If adding a kernel or optimization, confirm numerical equivalence -->
- [ ] Results validated against PyTorch reference implementation
- [ ] Absolute tolerance tested: `torch.allclose(triton_out, pytorch_out, atol=1e-2)`

## 📚 Related Issues / Papers
<!-- Link to GitHub issues and papers -->
- Closes #
- Ref: 

## 📝 Checklist
- [ ] PR title follows `type(scope): description` convention
- [ ] Branch target is `develop` (not `main`)
- [ ] No unrelated changes included
