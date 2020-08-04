import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy as scientropy

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


class EntropyMemory:
    """Estimate policy entropy."""
    def __init__(self, intial_bins=None, initial_count=1, base=None):
        # Init the count model
        if intial_bins is None:
            self.N = 1
        else:
            self.N = len(intial_bins)

        self.base = base
        self.initial_count = initial_count
        self.memory = dict()

        # Preinit its values?
        if intial_bins is not None:
            for x in intial_bins:
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
    """Bandit agent - sample(R + beta H(actions)"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    default_reward_value = 0  # Null R
    R_t = default_reward_value

    # Agents and memories
    critic = Critic(num_actions, default_value=default_reward_value)
    actor = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    all_actions = list(range(num_actions))

    entropy = EntropyMemory(intial_bins=all_actions, initial_count=1, base=2)

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
        entropy_bonus = entropy(action)
        print(entropy.probs)
        R_t += beta * entropy_bonus

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("bonus", entropy_bonus, n)
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
                  entropy=entropy.state_dict(),
                  total_R=total_R,
                  total_regret=total_regret,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)