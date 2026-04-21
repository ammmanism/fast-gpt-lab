import pytest
from src.vanilla.tensor_cores import enforce_tensor_core_alignment, pad_vocab_size

def test_tensor_core_validation_rejection():
    # GPT-2 original vocab size is notoriously unaligned
    vocab = 50257
    hidden = 768
    head = 64
    
    assert enforce_tensor_core_alignment(vocab, hidden, head) is False

def test_tensor_core_validation_success():
    # Padded vocab
    vocab = 50264
    # Unconventional hidden dim that is still perfectly aligned to 8
    hidden = 1000
    head = 120
    
    assert enforce_tensor_core_alignment(vocab, hidden, head) is True

def test_vocab_padding_logic():
    original = 50257
    padded = pad_vocab_size(original, alignment=8)
    
    # Needs to jump to 50264
    assert padded == 50264
    assert padded % 8 == 0
    
    # If already aligned, should not change
    assert pad_vocab_size(50264) == 50264
