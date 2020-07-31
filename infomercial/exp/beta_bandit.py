import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from collections import OrderedDict
from scipy.stats import entropy
from copy import deepcopy

from infomercial.distance import kl
from infomercial.memory import Count
from infomercial.utils import estimate_regret
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint


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
    def __init__(self, num_actions, tie_break='next', tie_threshold=0.0):
        self.num_actions = num_actions
        self.tie_break = tie_break
        self.tie_threshold = tie_threshold
        self.action_count = 0

    def _is_tied(self, values):
        # One element can't be a tie
        if len(values) < 1:
            return False

        # Apply the threshold, rectifying values less than 0
        t_values = [max(0, v - self.tie_threshold) for v in values]

        # Check for any difference, if there's a difference then
        # there can be no tie.
        tied = True  # Assume tie
        v0 = t_values[0]
        for v in t_values[1:]:
            if np.isclose(v0, v):
                continue
            else:
                tied = False

        return tied

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Pick the best as the base case, ....
        action = np.argmax(values)

        # then check for ties.
        #
        # Using the first element is argmax's tie breaking strategy
        if self.tie_break == 'first':
            pass
        # Round robin through the options for each new tie.
        elif self.tie_break == 'next':
            self.tied = self._is_tied(values)
            if self.tied:
                self.action_count += 1
                action = self.action_count % self.num_actions
        else:
            raise ValueError("tie_break must be 'first' or 'next'")

        return action


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditOneHigh2-v0',
        num_episodes=1,
        tie_break='next',
        tie_threshold=0.0,
        beta=1.0,
        lr_R=.1,
        master_seed=42,
        log_dir=None,
        write_to_disk=True):
    """Bandit agent - argmax(R + beta E)"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    # -
    default_reward_value = 0  # Null R
    default_info_value = entropy(np.ones(num_actions) /
                                 num_actions)  # Uniform p(a)
    E_t = default_info_value
    R_t = default_reward_value

    # Agents and memories
    critic = Critic(num_actions,
                    default_value=default_reward_value +
                    (beta * default_info_value))
    actor = Actor(num_actions,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold)

    memories = [Count() for _ in range(num_actions)]
    all_actions = list(range(num_actions))

    # -
    total_R = 0.0
    total_E = 0.0
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

        # Estimate E
        old = deepcopy(memories[action])
        memories[action].update((int(state), int(R_t)))
        new = deepcopy(memories[action])
        E_t = kl(new, old, default_info_value)

        # Critic learns
        critic = Q_update(action, R_t + (beta * E_t), critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_ER", critic(action), n)

        total_E += E_t
        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_E", total_E, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

        tie = 0
        if actor.tied:
            tie = 1
        writer.add_scalar("ties", tie, n)

    # -- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=env.best,
                  beta=beta,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold,
                  critic=critic.state_dict(),
                  total_E=total_E,
                  total_R=total_R,
                  total_regret=total_regret,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)