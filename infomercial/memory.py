import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from collections import deque

from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as scientropy

import random


class NoveltyMemory:
    def __init__(self, bonus=0):
        self.bonus = bonus
        self.memory = []

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        if state in self.memory:
            bonus = 0
        else:
            self.memory.append(state)
            bonus = self.bonus

        return bonus

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class CountMemory:
    """A simple state counter."""
    def __init__(self):
        self.memory = dict()

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        # Init?
        if state not in self.memory:
            self.memory[state] = 0

        # Update count in memory
        # and then return it
        self.memory[state] += 1

        return self.memory[state]

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class EntropyMemory:
    """Estimate policy entropy."""
    def __init__(self, initial_bins=None, initial_count=1, base=None):
        # Init the count model
        if initial_bins is None:
            self.N = 1
        else:
            self.N = len(initial_bins)

        self.base = base
        self.initial_count = initial_count
        self.memory = dict()

        # Preinit its values?
        if initial_bins is not None:
            for x in initial_bins:
                self.memory[x] = self.initial_count

    def __call__(self, action):
        return self.forward(action)

    def forward(self, action):
        # Init?
        if action not in self.memory:
            self.memory[action] = self.initial_count

        # Update count in memory
        self.N += 1
        self.memory[action] += 1

        # Estimate H
        self.probs = [(n / self.N) for n in self.memory.values()]
        return scientropy(np.asarray(self.probs), base=self.base)

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class ModulusMemory:
    """A very generic memory system, with a finite capacity."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory = []
        self.position = 0

    def state_dict(self):
        return {'position': self.position, 'memory': self.memory}

    def load_state_dict(self, state_dict):
        self.memory = state_dict['memory']
        self.position = state_dict['position']

    def __len__(self):
        return len(self.memory)


class DiscreteDistribution:
    """A discrete distribution."""
    def __init__(self, initial_bins=None, initial_count=1):
        # Init the count model
        self.N = 0
        self.initial_count = initial_count
        self.count = OrderedDict()

        # Preinit its values?
        if initial_bins is not None:
            for x in initial_bins:
                self.count[x] = self.initial_count

    def __len__(self):
        return len(self.count)

    def __call__(self, x):
        return self.forward(x)

    def update(self, x):
        # Init, if necessary
        if x not in self.count:
            self.count[x] = self.initial_count

        # Update the counts
        self.count[x] += 1
        self.N += 1

    def forward(self, x):
        if x not in self.count:
            return 0
        elif self.N == 0:
            return 0
        else:
            return self.count[x] / self.N

    def keys(self):
        return list(self.count.keys())

    def values(self):
        return list(self.count.values())

    def state_dict(self):
        return self.count

    def load_state_dict(self, state_dict):
        self.count = state_dict


class Kernel:
    """A continous distribution, estimated using a kernel
    
    NOTE: This is a thin wrapper around KernelDensity from the sklearn 
    library.
    
    For information on its hyperparamers see,
    https://scikit-learn.org/stable/modules/density.html#kernel-density
    """
    def __init__(self, **kernel_kwargs):
        self.dist = KernelDensity(**kernel_kwargs)
        self.X = []

    def update(self, x):
        self.X.append(x)
        self.Xvec = np.vstack(self.X)

        # X -> Xvec : (n_samples, n_features)
        # per sklearn standard shape
        if self.Xvec.ndim == 1:
            self.Xvec = np.expand_dims(self.Xvec, 1)
        elif self.Xvec.ndim > 2:
            raise ValueError("x must be a scalar or 1d list/array.")
        else:
            pass

        # Refit the dist over all the data seen so far; this must
        # be done w/ each new sample. Not eff. But this is a limit
        # of the sklearn API (it seems).
        self.dist.fit(self.Xvec)

    def forward(self, x):
        # Data can be a scalar or a list.
        # Reshape it to match the expected (1, n_feature) where '1' stands
        # in for 1 sample.
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)
        else:
            pass

        # Scores of log(p), but we want p.
        return float(np.exp(self.dist.score_samples(x)))


# # class ConditionalCount(Count):
# #     """A conditional discrete distribution."""
# #     def __init__(self):
# #         self.Ns = []
# #         self.conds = []
# #         self.counts = []

# #     def __call__(self, x, cond):
# #         return self.forward(x, cond)

# #     def keys(self, cond):
# #         if cond in self.conds:
# #             i = self.conds.index(cond)
# #             return list(self.counts[i].keys())
# #         else:
# #             return []

# #     def update(self, x, cond):
# #         # Add cond?
# #         if cond not in self.conds:
# #             self.conds.append(cond)
# #             self.counts.append(OrderedDict())
# #             self.Ns.append(0)

# #         # Locate cond.
# #         i = self.conds.index(cond)

# #         # Update counts for cond
# #         if x in self.counts[i]:
# #             self.counts[i][x] += 1
# #         else:
# #             self.counts[i][x] = 1

# #         # Update cond count normalizer
# #         self.Ns[i] += 1

# #     def forward(self, x, cond):
# #         # Locate cond.
# #         if cond not in self.conds:
# #             return 0
# #         else:
# #             i = self.conds.index(cond)

# #         # Get p(x|cond)
# #         if x not in self.counts[i]:
# #             return 0
# #         elif self.Ns[i] == 0:
# #             return 0
# #         else:
# #             return self.counts[i][x] / self.Ns[i]

# #     def probs(self, xs, conds):
# #         p = []
# #         for x, cond in zip(xs, conds):
# #             p.append(self.forward(x, cond))
# #         return p

# #     def values(self, xs, conds):
# #         return self.probs(xs, conds)

# class ConditionalMean(Memory):
#     """An averaging memory."""
#     def __init__(self):
#         self.conds = []
#         self.means = []
#         self.N = 1

#     def __call__(self, x, cond):
#         return self.forward(x, cond)

#     def update(self, x, cond):
#         # Add cond?
#         if cond not in self.conds:
#             self.conds.append(cond)
#             self.deltas.append(x)

#         # Locate cond.
#         i = self.conds.index(cond)

#         # Update the mean
#         delta = x - self.means[i]
#         self.means[i] += delta / self.N

#         # Update count
#         self.N += 1

#     def forward(self, x, cond):
#         # Locate cond.
#         if cond not in self.conds:
#             return 0
#         else:
#             i = self.conds.index(cond)

#         # Get the mean
#         return self.means[i]

#     def values(self, xs, conds):
#         p = []
#         for x, cond in zip(xs, conds):
#             p.append(self.forward(x, cond))
#         return p

# class ConditionalDeviance(Memory):
#     """A memory for deviance."""
#     def __init__(self):
#         self.mean = ConditionalMean()

#     def __call__(self, x, cond):
#         return self.forward(x, cond)

#     def update(self, x, cond):
#         self.mean.update(x, cond)

#     def forward(self, x, cond):
#         return x - self.mean(x, cond)

#     def values(self, xs, conds):
#         p = []
#         for x, cond in zip(xs, conds):
#             p.append(self.forward(x, cond))
#         return p

# class ConditionalDerivative(Memory):
#     """A memory for change."""
#     def __init__(self, delta_t=1):
#         self.conds = []
#         self.deltas = []
#         self.delta_t = delta_t
#         if self.delta_t < 0:
#             raise ValueError("delta_t must be positive")

#     def __call__(self, x, cond):
#         return self.forward(x, cond)

#     def update(self, x, cond):
#         # Add cond?
#         if cond not in self.conds:
#             self.conds.append(cond)
#             self.deltas.append(x)

#         # Locate cond.
#         i = self.conds.index(cond)

#         # Update counts for cond
#         self.deltas[i] = x - self.deltas[i]

#     def forward(self, x, cond):
#         # Locate cond.
#         if cond not in self.conds:
#             return 0
#         else:
#             i = self.conds.index(cond)

#         # Est. the dirative
#         return self.deltas[i] / self.delta_t

#     def values(self, xs, conds):
#         p = []
#         for x, cond in zip(xs, conds):
#             p.append(self.forward(x, cond))
#         return p

# class EfficientConditionalCount(Memory):
#     """Forget x when over-capacity"""
#     def __init__(self, capacity=1):
#         if capacity < 1:
#             raise ValueError("capacity must be >= 1")
#         self.capacity = capacity
#         self.conds = []
#         self.datas = []

#     def __call__(self, x, cond):
#         return self.forward(x, cond)

#     def update(self, x, cond):
#         # Add cond?
#         if cond not in self.conds:
#             self.conds.append(cond)
#             self.datas.append(deque(maxlen=self.capacity))

#         # Locate cond.
#         i = self.conds.index(cond)

#         # Update
#         self.datas[i].append(x)

#     def forward(self, x, cond):
#         # Locate cond.
#         if cond not in self.conds:
#             return 0
#         else:
#             i = self.conds.index(cond)

#         count = self.datas[i].count(x)
#         return count / self.capacity

#     def probs(self, xs, conds):
#         p = []
#         for x, cond in zip(xs, conds):
#             p.append(self.forward(x, cond))

#         return p

#     def values(self, xs, conds):
#         return self.probs(xs, conds)

# class ForgetfulConditionalCount(Memory):
#     """Forget conditions when over-capacity"""
#     def __init__(self, capacity=1):
#         if capacity < 1:
#             raise ValueError("capacity must be >= 1")

#         self.capacity = capacity
#         self.Ns = deque(maxlen=self.capacity)
#         self.conds = deque(maxlen=self.capacity)
#         self.counts = deque(maxlen=self.capacity)

#     def __call__(self, x, cond):
#         return self.forward(x, cond)

#     def update(self, x, cond):
#         # Add cond?
#         if cond not in self.conds:
#             self.conds.append(cond)
#             self.counts.append(OrderedDict())
#             self.Ns.append(0)

#         # Locate cond.
#         i = self.conds.index(cond)

#         # Update counts for cond
#         if x in self.counts[i]:
#             self.counts[i][x] += 1
#         else:
#             self.counts[i][x] = 1

#         # Update cond count normalizer
#         self.Ns[i] += 1

#     def forward(self, x, cond):
#         # Locate cond.
#         if cond not in self.conds:
#             return 0
#         else:
#             i = self.conds.index(cond)

#         # Get p(x|cond)
#         if x not in self.counts[i]:
#             return 0
#         elif self.Ns[i] == 0:
#             return 0
#         else:
#             return self.counts[i][x] / self.Ns[i]

#     def probs(self, xs, conds):
#         p = []
#         for x, cond in zip(xs, conds):
#             p.append(self.forward(x, cond))
#         return p

#     def values(self, xs, conds):
#         return self.probs(xs, conds)

# ----------------------------------------------------------------------------
# """
# TODO: move to Memory API; how to train/update this VAE?

# Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
# Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
# """

# class MaskedLinear(nn.Linear):
#     """ same as Linear except has a configurable mask on the weights """
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__(in_features, out_features, bias)
#         self.register_buffer('mask', torch.ones(out_features, in_features))

#     def set_mask(self, mask):
#         self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

#     def forward(self, input):
#         return F.linear(input, self.mask * self.weight, self.bias)

# class MADE(nn.Module):
#     def __init__(self,
#                  nin,
#                  hidden_sizes,
#                  nout,
#                  num_masks=1,
#                  natural_ordering=False):
#         """
#         nin: integer; number of inputs
#         hidden sizes: a list of integers; number of units in hidden layers
#         nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
#               note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
#               will be all the means and the second nin will be stds. i.e. output dimensions depend on the
#               same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
#               the output of running the tests for this file makes this a bit more clear with examples.
#         num_masks: can be used to train ensemble over orderings/connections
#         natural_ordering: force natural ordering of dimensions, don't use random permutations
#         """

#         super().__init__()
#         self.nin = nin
#         self.nout = nout
#         self.hidden_sizes = hidden_sizes
#         assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

#         # define a simple MLP neural net
#         self.net = []
#         hs = [nin] + hidden_sizes + [nout]
#         for h0, h1 in zip(hs, hs[1:]):
#             self.net.extend([
#                 MaskedLinear(h0, h1),
#                 nn.ReLU(),
#             ])
#         self.net.pop()  # pop the last ReLU for the output layer
#         self.net = nn.Sequential(*self.net)

#         # seeds for orders/connectivities of the model ensemble
#         self.natural_ordering = natural_ordering
#         self.num_masks = num_masks
#         self.seed = 0  # for cycling through num_masks orderings

#         self.m = {}
#         self.updatemasks()  # builds the initial self.m connectivity
#         # note, we could also precompute the masks and cache them, but this
#         # could get memory expensive for large number of masks.

#     def updatemasks(self):
#         if self.m and self.num_masks == 1:
#             return  # only a single seed, skip for efficiency
#         L = len(self.hidden_sizes)

#         # fetch the next seed and construct a random stream
#         rng = np.random.RandomState(self.seed)
#         self.seed = (self.seed + 1) % self.num_masks

#         # sample the order of the inputs and the connectivity of all neurons
#         self.m[-1] = np.arange(
#             self.nin) if self.natural_ordering else rng.permutation(self.nin)
#         for l in range(L):
#             self.m[l] = rng.randint(self.m[l - 1].min(),
#                                     self.nin - 1,
#                                     size=self.hidden_sizes[l])

#         # construct the mask matrices
#         masks = [
#             self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)
#         ]
#         masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

#         # handle the case where nout = nin * k, for integer k > 1
#         if self.nout > self.nin:
#             k = int(self.nout / self.nin)
#             # replicate the mask across the other outputs
#             masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

#         # set the masks in all MaskedLinear layers
#         layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
#         for l, m in zip(layers, masks):
#             l.set_mask(m)

#     def forward(self, x):
#         return self.net(x)
