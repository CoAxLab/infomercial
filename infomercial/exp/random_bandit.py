import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from scipy.stats import entropy
from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint

from collections import OrderedDict


class Critic(object):
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


class Actor(object):
    def __init__(self, num_actions, seed_value=42):
        self.num_actions = num_actions
        self.seed_value = seed_value
        self.prng = np.random.RandomState(self.seed_value)

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        action = self.prng.randint(0, self.num_actions, size=1)[0]

        return action


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None):
    """Bandit agent - random"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    # -
    default_reward_value = 0  # Null R
    R_t = default_reward_value
    critic = Critic(num_actions, default_value=default_reward_value)
    actor = Actor(num_actions, seed_value=master_seed)
    all_actions = list(range(num_actions))

    # ------------------------------------------------------------------------
    num_best = 0
    total_R = 0.0
    total_regret = 0.0

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

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
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
                  env_name=env_name,
                  num_episodes=num_episodes,
                  lr_R=lr_R,
                  critic=critic.state_dict(),
                  total_R=total_R,
                  total_regret=total_regret,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)