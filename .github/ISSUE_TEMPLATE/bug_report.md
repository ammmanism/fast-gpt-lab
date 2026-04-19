---
name: 🐛 Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] <short description>'
labels: ['bug', 'triage']
assignees: ''
---

## 📋 Bug Description
A clear and concise description of the bug.

## 🔁 Minimal Reproducible Example
```python
# Paste minimal code that reproduces the issue
from src.vanilla.config import GPTConfig
from src.vanilla.model import GPT
...
```

## 📊 Expected vs Actual Behavior
| | Expected | Actual |
|---|---|---|
| Behavior | ... | ... |
| Loss value | ... | ... |
| MFU | ... | ... |

## 🖥️ Environment
- **GPU**: `nvidia-smi | head -2`
- **CUDA**: `nvcc --version`
- **PyTorch**: `python -c "import torch; print(torch.__version__)"`
- **Triton**: `python -c "import triton; print(triton.__version__)"`
- **OS**: 
- **Python**: 

## 📎 Additional Context
- Training config (paste relevant `TrainConfig` / `GPTConfig` fields)
- Stack trace:
```
Paste full traceback here
```
