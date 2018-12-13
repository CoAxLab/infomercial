import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal


class Memory(object):
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

    def __len__(self):
        return len(self.memory)


class LinearCategorical(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super(LinearCategorical, self).__init__()
        self.affine1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.affine1(x)
        return F.softmax(x, dim=1)
