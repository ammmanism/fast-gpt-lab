#!/bin/bash
# ─── A100 Profiling Commands — fast-gpt-lab ───────────────────────────────────
# Run these on an A100 SXM4 to profile kernel performance.
# Outputs saved to benchmarks/results/

set -euo pipefail

RESULTS_DIR="benchmarks/results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "A100 Profiling Suite — fast-gpt-lab"
echo "=========================================="

# ─── NSYS: System-wide timeline analysis ──────────────────────────────────────
# Captures kernel launches, CPU-GPU synchronization, memory transfers, and timeline.
echo ""
echo "[1/3] Running nsys profile (system timeline)..."
nsys profile \
    --stats=true \
    --force-overwrite=true \
    --output="$RESULTS_DIR/a100_nsys_report" \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python benchmarks/run_mfu_calculator.py

echo "nsys report saved to $RESULTS_DIR/a100_nsys_report.qdrep"
echo "  View with: nsys-ui $RESULTS_DIR/a100_nsys_report.qdrep"

# ─── NCU: Kernel-level detailed analysis ──────────────────────────────────────
# Captures SM occupancy, memory throughput, Tensor Core utilization, and roofline.
echo ""
echo "[2/3] Running ncu profile (kernel detail) for FlashAttention..."
ncu \
    --set full \
    --target-processes all \
    --kernel-name "flash_attention" \
    --launch-count 1 \
    --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        l2_cache__throughput.avg.pct_of_peak_sustained_elapsed,\
        smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
        achieved_occupancy,\
        theoretical_occupancy,\
        elapsed_cycles \
    --output "$RESULTS_DIR/a100_ncu_flash_attention" \
    python benchmarks/run_micro_bench.py 2>&1 | tee "$RESULTS_DIR/a100_ncu_flash_attention.log"

echo "NCU FlashAttention report saved to $RESULTS_DIR/a100_ncu_flash_attention.log"

echo ""
echo "[3/3] Running ncu profile (kernel detail) for SwiGLU..."
ncu \
    --set full \
    --target-processes all \
    --kernel-name "swiglu" \
    --launch-count 1 \
    --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
        achieved_occupancy,\
        elapsed_cycles \
    --output "$RESULTS_DIR/a100_ncu_swiglu" \
    python -c "
from src.kernels.swiglu import fused_swiglu
import torch
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
w_gate = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
w_up = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(10): fused_swiglu(x, w_gate, w_up)
" 2>&1 | tee "$RESULTS_DIR/a100_ncu_swiglu.log"

echo "NCU SwiGLU report saved to $RESULTS_DIR/a100_ncu_swiglu.log"

# ─── QUICK NCU SUMMARY ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Quick NCU Summary (FlashAttention)"
echo "=========================================="
ncu --query-metrics --kernel-name "flash_attention" --log-file "$RESULTS_DIR/a100_ncu_flash_attention.log" 2>/dev/null | head -50

echo ""
echo "=========================================="
echo "Profiling Complete"
echo "=========================================="
echo "Artifacts:"
echo "  - $RESULTS_DIR/a100_nsys_report.qdrep (open in nsys-ui)"
echo "  - $RESULTS_DIR/a100_ncu_flash_attention.log"
echo "  - $RESULTS_DIR/a100_ncu_swiglu.log"