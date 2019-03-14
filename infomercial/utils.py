import torch
import numpy as np
from collections import OrderedDict


class Hyperparameters(object):
    """A hyperparameters holder."""

    def __init__(self):
        pass

    def param_dict(self):
        return self.__dict__


def build_hyperparameters(**kwargs):
    """Build a Hyperparameters instance"""

    hp = Hyperparameters()
    for k, v in kwargs.items():
        setattr(hp, k, v)

    return hp


def create_envs(env_name, num_processes, hp):
    # Setup vec env....
    env_list = [
        make_env_vec(env_name, hp.seed_value, i) for i in range(num_processes)
    ]
    envs = gym_vecenv.DummyVecEnv(env_list)
    if num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(env_list)
    if len(envs.observation_space.shape) == 1:
        envs = gym_vecenv.VecNormalize(envs)

    return envs


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


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape