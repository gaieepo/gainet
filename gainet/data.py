import numpy as np
from .tensor import Tensor
from collections import namedtuple
from typing import NamedTuple

Batch = NamedTuple('Batch', [('xs', Tensor), ('ys', Tensor)])


class DataIterator:
    def __call__(self, xs):
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, xs, ys):
        starts = np.arange(0, len(xs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_xs = xs[start:end]
            batch_ys = ys[start:end]
            yield Batch(batch_xs, batch_ys)
