import numpy as np
from collections import OrderedDict


class Distribution(object):
    """Estimate a discrete distribution."""

    def __init__(self):
        self.N = 0.0
        self.counts = OrderedDict()

    def update(self, state):
        if state in self.counts:
            self.counts[state] += 1
        else:
            self.counts[state] = 1

    def keys(self):
        return list(self.counts.keys())

    def values(self):
        return [v / self.N for v in self.counts.values()]

    def items(self):
        k = self.keys()
        v = self.values()
        return zip(k, v)

    def __call__(self, state):
        return self.counts[state] / self.N
