import torch
import numpy as np
from collections import OrderedDict


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Distribution(object):
    """Estimate a discrete distribution."""

    def __init__(self):
        self.N = 0
        self.counts = OrderedDict()

    def update(self, state):
        if state in self.counts:
            self.counts[state] += 1
        else:
            self.counts[state] = 1
        self.N += 1

    def keys(self):
        return list(self.counts.keys())

    def values(self):
        if self.N > 0:
            return [v / self.N for v in self.counts.values()]
        else:
            return [0] * len(self.counts)

    def items(self):
        k = self.keys()
        v = self.values()
        return zip(k, v)

    def __call__(self, state):
        return self.counts[state] / self.N
