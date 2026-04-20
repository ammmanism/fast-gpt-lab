import pytest
from src.vanilla.data_sharder import DistDataSharder

def test_data_sharding_distribution():
    base_iterator = iter(range(10))
    # Rank 0 in a 2-GPU node
    sharded_0 = DistDataSharder(base_iterator, rank=0, world_size=2)
    
    assert next(sharded_0) == 0
    # Next element in base is 1, but modulo is for rank 1, so our sharder skips to 2
    assert next(sharded_0) == 2
    assert next(sharded_0) == 4

def test_data_sharding_mutually_exclusive():
    world_size = 3
    results = {0: [], 1: [], 2: []}
    
    # Simulate parallel workers consuming the independent copies of the stream
    for rank in range(world_size):
        stream = iter(range(12))
        sharder = DistDataSharder(stream, rank=rank, world_size=world_size)
        results[rank] = list(sharder)
        
    assert results[0] == [0, 3, 6, 9]
    assert results[1] == [1, 4, 7, 10]
    assert results[2] == [2, 5, 8, 11]
