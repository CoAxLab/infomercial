import fire
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

from infomercial import models
from infomercial.models import Memory
from infomercial.util import save_checkpoint

EPS = np.finfo(np.float32).eps.item()


def select_action(policy, state, mode='Categorical'):
    # Get the current policy pi_s
    state = state.float().unsqueeze(0)
    pi_s = policy(state)

    # Use pi_s to make an action, using any dist in torch.
    # The dist should match the policy, of course.
    Dist = getattr(torch.distributions, mode)
    m = Dist(*pi_s)
    action = m.sample()

    log_prob = m.log_prob(action).unsqueeze(0)
    return policy, action.item(), log_prob


def train():
    pass
