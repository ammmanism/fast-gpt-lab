import pytest
from src.tokenizer.bpe import BPETokenizer

def test_bpe_training_and_encoding():
    tok = BPETokenizer()
    corpus = "hello world, hello universe. the quick brown fox jumps over the lazy dog."
    
    # Train a very small vocabulary
    tok.train(corpus, vocab_size=260, verbose=False)
    assert len(tok) >= 260
    
    # Test encoding and decoding idempotence
    test_str = "hello world"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    
    assert decoded == test_str, f"Expected '{test_str}', got '{decoded}'"

def test_special_tokens():
    tok = BPETokenizer()
    eot = tok.add_special_token("<|endoftext|>")
    
    assert "<|endoftext|>" in tok.vocab
    assert tok.vocab["<|endoftext|>"] == eot
