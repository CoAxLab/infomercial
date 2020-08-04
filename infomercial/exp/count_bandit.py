import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy

from collections import OrderedDict

from infomercial.utils import estimate_regret
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint
from infomercial.distance import kl


class Critic:
    def __init__(self, num_inputs, default_value):
        self.num_inputs = num_inputs
        self.default_value = default_value

        self.model = OrderedDict()
        for n in range(self.num_inputs):
            self.model[n] = self.default_value

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        return self.model[state]

    def update_(self, state, update):
        self.model[state] += update

    def state_dict(self):
        return self.model


class SoftmaxActor:
    def __init__(self, num_actions, temp=1, seed_value=42):
        self.temp = temp
        self.num_actions = num_actions
        self.seed_value = seed_value
        self.prng = np.random.RandomState(self.seed_value)
        self.actions = list(range(self.num_actions))

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Convert to ps
        values = np.asarray(values)
        z = values * (1 / self.temp)
        x = np.exp(z)
        ps = x / np.sum(x)

        # Sample actions by ps
        action = self.prng.choice(self.actions, p=ps)

        return action


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


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditOneHigh2-v0',
        num_episodes=1,
        tie_threshold=0.0,
        temp=1.0,
        beta=1.0,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None):
    """Bandit agent - sample(R + beta count^(-1/2))"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    default_reward_value = 0  # Null R
    default_info_value = entropy(np.ones(num_actions) /
                                 num_actions)  # Uniform p(a)
    E_t = default_info_value
    R_t = default_reward_value

    # Agents and memories
    critic = Critic(num_actions,
                    default_value=default_reward_value +
                    (beta * default_info_value))
    actor = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    all_actions = list(range(num_actions))

    count = CountMemory()

    # -
    total_R = 0.0
    total_regret = 0.0
    num_best = 0

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        env.reset()

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regret = estimate_regret(all_actions, action, critic)

        # Pull a lever.
        state, R_t, _, _ = env.step(action)

        # Apply count bonus
        count_bonus = count(state)**(-0.5)
        R_t += beta * count_bonus

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("bonus", count_bonus, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_R", critic(action), n)

        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

    # -- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=env.best,
                  beta=beta,
                  temp=temp,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  tie_threshold=tie_threshold,
                  critic=critic.state_dict(),
                  count=count.state_dict(),
                  total_R=total_R,
                  total_regret=total_regret,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)