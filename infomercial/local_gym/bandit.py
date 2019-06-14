#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

# Gym is annoying these days...
import warnings
warnings.filterwarnings("ignore")


class BanditEnv(gym.Env):
    """
    n-armed bandit environment  

    Params
    ------
    p_dist : list
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist : list or list or lists
        A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has
    """

    def __init__(self, p_dist, r_dist):
        if len(p_dist) != len(r_dist):
            raise ValueError(
                "Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError(
                    "Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(self.n_bandits)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        state = action

        reward = 0
        done = True

        if self.np_random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = self.np_random.normal(self.r_dist[action][0],
                                               self.r_dist[action][1])

        return [0], reward, done, {}

    def reset(self):
        return [0]

    def render(self, mode='human', close=False):
        pass


class UnstableBanditEnv(gym.Env):
    """n-armed bandit, but the winning probabilites are unstable."""

    def __init__(self, p_dists, r_dists, unstable_rate):
        for p_dist, r_dist in zip(p_dists, r_dists):
            if len(p_dist) != len(r_dist):
                raise ValueError(
                    "Probability and Reward distribution must be the same length"
                )

            if min(p_dist) < 0 or max(p_dist) > 1:
                raise ValueError("All probabilities must be between 0 and 1")

            for reward in r_dist:
                if isinstance(reward, list) and reward[1] <= 0:
                    raise ValueError(
                        "Standard deviation in rewards must all be greater than 0"
                    )

        # Set up first dist
        self.p_dists = p_dists
        self.r_dists = r_dists
        self._random_bandit()

        self.unstable_rate = unstable_rate

        # Setup the space
        self.n_bandits = len(self.p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(self.n_bandits)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _random_bandit(self):
        i = self.np_random.randint(0, len(self.p_dists))
        self.p_dist = self.p_dists[i]
        self.r_dist = self.r_dists[i]

    def step(self, action):
        assert self.action_space.contains(action)
        state = action

        reward = 0
        done = True

        if self.np_random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = self.np_random.normal(self.r_dist[action][0],
                                               self.r_dist[action][1])

        # Change bandit?
        if self.np_random.poisson(self.unstable_rate) > 0:
            self._random_bandit()

        return [0], reward, done, {}

    def reset(self):
        return [0]

    def render(self, mode='human', close=False):
        pass


class BanditOneHot2(BanditEnv):
    """A one winner bandit."""

    def __init__(self):
        self.best = [0]
        self.num_arms = 2

        p_dist = [0] * self.num_arms
        p_dist[self.best[0]] = 1
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHot10(BanditEnv):
    """A one winner bandit."""

    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        p_dist = [0] * self.num_arms
        p_dist[self.best[0]] = 1
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHot121(BanditEnv):
    """A one winner bandit."""

    def __init__(self):
        self.best = [54]
        self.num_arms = 121

        p_dist = [0] * self.num_arms
        p_dist[self.best[0]] = 1
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHot1000(BanditEnv):
    """A one winner bandit."""

    def __init__(self):
        self.best = [526]
        self.num_arms = 1000

        p_dist = [0] * self.num_arms
        p_dist[self.best[0]] = 1
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditEvenOdds2(BanditEnv):
    """A 50/50 bandit."""

    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.5, 0.5], r_dist=[1, 1])


class BanditOneHigh2(BanditEnv):
    """A (0.8, 0.2) bandit."""

    def __init__(self):
        self.best = [0]
        self.num_arms = 2

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHigh10(BanditEnv):
    """A (0.8, 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHigh121(BanditEnv):
    """A (0.8, 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [54]
        self.num_arms = 121

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTwoHigh10(BanditEnv):
    """A (..., 0.80, ..., 0.80. 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [7, 3]
        self.num_arms = 10

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.40
        p_dist[self.best[1]] = 0.80
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTwoHigh121(BanditEnv):
    """A (..., 0.80, ..., 0.80. 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [26, 71]
        self.num_arms = 121

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.40
        p_dist[self.best[1]] = 0.80
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditOneHigh1000(BanditEnv):
    """A (0.8, 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [526]
        self.num_arms = 1000

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTwoHigh1000(BanditEnv):
    """A (..., 0.80, ..., 0.80. 0.2, 0.2, ...) bandit."""

    def __init__(self):
        self.best = [526, 731]
        self.num_arms = 1000

        p_dist = [0.2] * self.num_arms
        p_dist[self.best[0]] = 0.40
        p_dist[self.best[1]] = 0.80
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTwoExtreme1000(BanditEnv):
    """A (..., 0.99, ..., 0.99. 0.01, 0.01, ...) bandit."""

    def __init__(self):
        self.best = [526, 731]
        self.num_arms = 1000

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.99
        p_dist[self.best[1]] = 0.99
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse2(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""

    def __init__(self):
        self.best = [0]
        self.num_arms = 2

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse10(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""

    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse121(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""

    def __init__(self):
        self.best = [54]
        self.num_arms = 121

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditHardAndSparse1000(BanditEnv):
    """A (0.10,0.08,0.08,....) bandit"""

    def __init__(self):
        self.best = [526]
        self.num_arms = 1000

        p_dist = [0.01] * self.num_arms
        p_dist[self.best[0]] = 0.02
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditUniform10(BanditEnv):
    """A U(0.2, 0.75) bandit, with one best set 0.8."""

    def __init__(self):
        self.best = [7]
        self.num_arms = 10

        p_dist = np.random.uniform(0.2, 0.75, size=self.num_arms).tolist()
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditUniform121(BanditEnv):
    """A U(0.2, 0.75) bandit, with one best set 0.8."""

    def __init__(self):
        self.best = [54]
        self.num_arms = 121

        p_dist = np.random.uniform(0.2, 0.75, size=self.num_arms).tolist()
        p_dist[self.best[0]] = 0.8
        r_dist = [1] * self.num_arms
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditGaussian10(BanditEnv):
    """
    10 armed bandit mentioned on page 30 of Sutton and Barto's
    [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)

    Actions always pay out.
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)
    """

    def __init__(self, bandits=10):
        p_dist = np.full(bandits, 1)
        r_dist = []

        for i in range(bandits):
            r_dist.append([np.random.normal(0, 1), 1])

        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)