import math
import cloudpickle
import torch
import numpy as np

from collections import OrderedDict


def save_checkpoint(state, filename='checkpoint.pkl'):
    data = cloudpickle.dumps(state)
    with open(filename, 'wb') as fi:
        fi.write(data)


def load_checkpoint(filename='checkpoint.pkl'):
    with open(filename, 'rb') as fi:
        return cloudpickle.load(fi)


def sample_action(policy, state, mode='Categorical'):
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


def estimate_regret(states, state, critic):
    """Estimate regret, given a critic."""

    values = [critic(s) for s in states]
    best = np.max(values)
    actual = values[state]

    return best - actual


class Oracle(object):
    def __init__(self, env, step_value=1):
        """An oracle that counts ALL the steps taken in an Gym env."""

        self._env = env
        self.total_steps = 0
        self.step_value = step_value

    def __getattr__(self, attr):
        # NOTE: do not use hasattr, it goes into
        # infinite recurrsion

        # See if this object has attr
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the env
        return getattr(self._env, attr)

    def step(self, action):
        self.total_steps += self.step_value
        ret = self._env.step(action)
        return ret
