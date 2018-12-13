import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal


class LinearCategorical(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super(LinearCategorical, self).__init__()
        self.affine1 = nn.Linear(in_features, out_features)

        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        return F.softmax(x, dim=1)
