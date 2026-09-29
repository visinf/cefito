"""Evaluation sharding sampler."""

from torch.utils.data import Sampler


class EvalShardSampler(Sampler):
    """A rank's slice of the split, with no padding and no overlap.

    ``DistributedSampler`` pads the index list up to a multiple of the world
    size so that every rank yields the same number of batches. That is what a
    DDP training step needs, but it is wrong for a metric: the padding repeats
    real samples, and a metric reduced as ``(sum, count)`` across ranks then
    scores those samples twice. On the CrossTask validation split (4531 videos)
    at world size 8 it would double-count 5 of them.

    Evaluation runs the unwrapped EMA copy, so there is no collective inside the
    loop and the shards are free to differ in length by one. The reduction in
    :func:`~cefito.engine.evaluator.evaluate` sums both the totals and the
    counts, so unequal shards weight correctly and the answer matches a single
    process exactly.
    """

    def __init__(self, dataset_size: int, rank: int, world_size: int):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside a world of {world_size}")
        self.indices = range(rank, dataset_size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
