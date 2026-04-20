import pytest
import torch
import unittest.mock as mock
from src.vanilla.streaming_data import StreamingDataLoader

@mock.patch("src.vanilla.streaming_data.load_dataset")
def test_streaming_loader_shapes(mock_load):
    # Mocking a trivial dataset response
    mock_load.return_value = [{"text": "hello world test sentence"}] * 100
    
    loader = StreamingDataLoader("dummy", batch_size=2, block_size=4, device="cpu")
    
    # Take first batch
    x, y = next(loader)
    
    # Assert dimensions
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    
    # Assert elements are tokens (integers)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
