from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, defaultdict
from collections import deque

from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as scientropy

import random


class NoveltyMemory:
    def __init__(self, bonus=0):
        self.bonus = bonus
        self.memory = dict()

    def __call__(self, state):
        return self.forward(state)

    def __len__(self):
        return len(self.memory)

    def forward(self, state):
        if state in self.memory:
            bonus = 0
        else:
            bonus = self.bonus
        return bonus

    def update(self, state):
        self.memory[state] = 1

    def keys(self):
        return list(self.memory.keys())

    def values(self):
        return list(self.memory.values())

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class RateMemory:
    """A finite memory of the reward rate changes."""
    def __init__(self, maxlen=10, default_value=0.0):
        self.maxlen = maxlen
        self.default_value = default_value
        self.memory = dict()

    def __call__(self, state):
        return self.forward(state)

    def __len__(self):
        return len(self.memory)

    def forward(self, state):
        rate = np.mean(self.memory[state])
        return rate

    def update(self, state, reward):
        # Init?
        if state not in self.memory:
            self.memory[state] = deque(maxlen=self.maxlen)
            for _ in range(self.maxlen):
                self.memory[state].append(self.default_value)

        # Update
        self.memory[state].append(reward)

    def keys(self):
        return list(self.memory.keys())

    def values(self):
        return list(self.memory.values())

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

    def __len__(self):
        return len(self.memory)

    def forward(self, state):
        return self.memory[state]

    def update(self, state):
        # Init?
        if state not in self.memory:
            self.memory[state] = 0

        # Update count in memory
        # and then return it
        self.memory[state] += 1

    def keys(self):
        return list(self.memory.keys())

    def values(self):
        return list(self.memory.values())

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

    def __len__(self):
        return len(self.memory)

    def forward(self, state):
        # Estimate H
        self.probs = [(n / self.N) for n in self.memory.values()]
        return scientropy(np.asarray(self.probs), base=self.base)

    def update(self, action):
        # Init?
        if action not in self.memory:
            self.memory[action] = self.initial_count

        # Update count in memory
        self.N += 1
        self.memory[action] += 1

    def keys(self):
        return list(self.memory.keys())

    def values(self):
        return list(self.memory.values())

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


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

    def forward(self, x):
        if x not in self.count:
            return 0
        elif self.N == 0:
            return 0
        else:
            return self.count[x] / self.N

    def update(self, x):
        # Init, if necessary
        if x not in self.count:
            self.count[x] = self.initial_count

        # Update the counts
        self.count[x] += 1
        self.N += 1

    def keys(self):
        return list(self.count.keys())

    def values(self):
        return list(self.count.values())

    def state_dict(self):
        return self.count

    def load_state_dict(self, state_dict):
        self.count = state_dict
