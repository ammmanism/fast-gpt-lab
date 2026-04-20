"""
Data Sharder — fast-gpt-lab
Ensures that each distributed worker processes mutually exclusive subsets of the data.
"""
from typing import Iterator

class DistDataSharder:
    """
    Wraps any python iterator to yield only the elements assigned to this rank.
    Vital for distributed data parallel logic so GPUs don't calculate identical gradients.
    """
    def __init__(self, iterator: Iterator, rank: int, world_size: int):
        self.iterator = iterator
        self.rank = rank
        self.world_size = world_size
        self._step = 0

    def __iter__(self):
        return self

    def __next__(self):
        # We must consume elements from the underlying iterator to advance it
        # but only yield the ones that match our rank modulo
        while True:
            item = next(self.iterator)
            is_our_turn = (self._step % self.world_size) == self.rank
            self._step += 1
            if is_our_turn:
                return item
