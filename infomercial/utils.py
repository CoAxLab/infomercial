import torch
import numpy as np
from collections import OrderedDict


def normal_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def best_action(policy, state):
    """Greedy action selection."""
    action = None
    return action


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def log_probability(x, mu, std, logstd):
    var = std.pow(2)
    log_density = (
        -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd)
    return log_density.sum(1, keepdim=True)


def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


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