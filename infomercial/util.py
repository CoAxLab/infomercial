import torch
import numpy as np
from collections import OrderedDict


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Distribution(object):
    """A discrete distribution."""

    def __init__(self):
        self.N = 0
        self.counts_data = OrderedDict()

    def update(self, state):
        if state in self.counts_data:
            self.counts_data[state] += 1
        else:
            self.counts_data[state] = 1
        self.N += 1

    def keys(self):
        return list(self.counts_data.keys())

    def probs(self):
        if self.N > 0:
            return [v / self.N for v in self.counts_data.values()]
        else:
            return [0] * len(self.counts_data)

    def counts(self):
        return self.counts_data.values()

    def items(self):
        k = self.keys()
        v = self.probs()
        return zip(k, v)

    def __call__(self, state):
        return self.counts_data[state] / self.N
