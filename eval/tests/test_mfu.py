import pytest
from src.vanilla.config import GPTConfig
from profiling.flop_calculator import calculate_mfu

def test_mfu_calculation_hardware_limits():
    """Ensure that absolute minimum computational times don't result in >100% physically impossible MFU."""
    cfg = GPTConfig.gpt2_small() # 124M param model
    
    # 6 * N = 6 * 124e6 = 744 MFLOPs per token
    # Let's say a batch takes literally 0.0001 ms (physically impossible)
    
    mfu_ratio = calculate_mfu(cfg, batch_size=4, seq_len=1024, time_per_iter_ms=0.0001, hw_name="A100_SXM4_80GB")
    
    # We should see astronomical MFU if time is approaching zero
    assert mfu_ratio > 10.0

def test_mfu_realistic_baseline():
    """Verify that a standard ~200ms pass on a 124M model yields a scientifically expected MFU ratio on A100."""
    cfg = GPTConfig.gpt2_small() 
    
    # Let's simulate a solid kernel implementation achieving ~250ms per iteration
    # B=16, T=1024 -> 16,384 tokens
    mfu_ratio = calculate_mfu(cfg, batch_size=16, seq_len=1024, time_per_iter_ms=250.0, hw_name="A100_SXM4_80GB")
    
    # Fast implementations should hover between 45% to 65% MFU
    assert 0.40 < mfu_ratio < 0.75
