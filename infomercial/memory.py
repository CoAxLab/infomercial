import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Memory(object):
    """Base Memory"""

    def update(self, x):
        """Update the Memory"""
        pass

    def forward(self, x):
        """p(x)"""
        pass

    def __call__(self, x):
        return self.forward(x)


class Count(Memory):
    """A discrete distribution."""

    def __init__(self):
        self.N = 0
        self.count = OrderedDict()

    def update(self, x):
        if x in self.count:
            self.count[x] += 1
        else:
            self.count[x] = 1
        self.N += 1

    def forward(self, x):
        if x not in self.count:
            return 0
        elif self.N == 0:
            return 0
        else:
            return self.count[x] / self.N

    def keys(self):
        return self.count.keys()

    def values(self):
        return self.count.values()


class ConditionalCount(Count):
    """A conditional discrete distribution."""

    def __init__(self):
        self.Ns = []
        self.conds = []
        self.counts = []

    def update(self, x, cond):
        # Add cond?
        if cond not in self.conds:
            self.conds.append(cond)
            self.counts.append(OrderedDict())
            self.Ns.append(0)

        # Locate cond.
        i = self.conds.index(cond)

        # Update counts for cond
        if x in self.counts[i]:
            self.counts[i][x] += 1
        else:
            self.counts[i][x] = 1

        # Update cond normalizer
        self.Ns[i] += 1

    def forward(self, x, cond):
        # Locate cond.
        if cond not in self.conds:
            return 0
        else:
            i = self.conds.index(cond)

        # Get p(x|cond)
        if x not in self.counts[i]:
            return 0
        elif self.Ns[i] == 0:
            return 0
        else:
            return self.counts[i][x] / self.Ns[i]


# ----------------------------------------------------------------------------
"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
"""


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self,
                 nin,
                 hidden_sizes,
                 nout,
                 num_masks=1,
                 natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(
            self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [
            self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)
        ]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)
