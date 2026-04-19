---
name: 🚀 Feature Request
about: Propose a new optimization, kernel, or evaluation
title: '[FEAT] <short description>'
labels: ['enhancement']
assignees: ''
---

## 💡 Feature Summary
One-line description of what you want to add.

## 🎯 Motivation
Which benchmark / metric does this improve? What's the expected gain?
- [ ] Throughput (tokens/sec / MFU)
- [ ] Memory efficiency (GB)
- [ ] Accuracy (perplexity / HellaSwag)
- [ ] Developer experience

## 📐 Theoretical Justification
Link to paper or derivation that motivates this feature.
> e.g. "RoPE embeddings (Su et al., 2021) enable length extrapolation at inference time..."

## 🛠️ Implementation Sketch
```python
# Pseudocode or key design decisions
```

## 🔬 Validation Plan
How will we know this works?
- [ ] Unit test in `eval/`
- [ ] Benchmark comparison in `benchmarks/`
- [ ] MFU delta in `profiling/`
- [ ] Perplexity on WikiText-103

## 📎 References
- Paper: 
- Implementation reference: 
