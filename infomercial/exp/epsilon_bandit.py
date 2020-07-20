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
    def __init__(self,
                 num_actions,
                 epsilon=0.1,
                 decay_tau=0.001,
                 seed_value=42):
        self.epsilon = epsilon
        self.decay_tau = decay_tau
        self.num_actions = num_actions
        self.seed_value = seed_value
        self.prng = np.random.RandomState(self.seed_value)

    def __call__(self, values):
        return self.forward(values)

    def decay_epsilon(self):
        self.epsilon -= (self.decay_tau * self.epsilon)

    def forward(self, values):
        # If values are zero, be random.
        if np.isclose(np.sum(values), 0):
            action = self.prng.randint(0, self.num_actions, size=1)[0]

            return action

        # Otherwise, do Ep greedy
        if self.prng.rand() < self.epsilon:
            action = self.prng.randint(0, self.num_actions, size=1)[0]
        else:
            action = np.argmax(values)

        return action


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        epsilon=0.1,
        epsilon_decay_tau=0,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None):
    """Play some slots!"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    # -
    default_reward_value = 0.0
    R_t = default_reward_value
    critic = Critic(env.observation_space.n,
                    default_value=default_reward_value)
    actor = Actor(num_actions,
                  epsilon=epsilon,
                  decay_tau=epsilon_decay_tau,
                  seed_value=master_seed)
    all_actions = list(range(num_actions))

    # -
    num_best = 0
    total_R = 0.0
    total_regret = 0.0

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regret = estimate_regret(all_actions, action, critic)

        # Pull a lever.
        state, R_t, _, _ = env.step(action)
        state = int(state[0])

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        writer.add_scalar("state", state, n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("epsilon", actor.epsilon, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_R", critic(action), n)

        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

        # Decay ep?
        if epsilon_decay_tau > 0:
            actor.decay_epsilon()

    # -- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=env.best,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  lr_R=lr_R,
                  epsilon=epsilon,
                  epsilon_decay_tau=epsilon_decay_tau,
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