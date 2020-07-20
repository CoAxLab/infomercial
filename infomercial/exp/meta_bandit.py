import os
import fire
import gym

import numpy as np

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy
from collections import OrderedDict

from infomercial.memory import Count
from infomercial.distance import kl
from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint


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

    def update_(self, state, update, replace=False):
        if replace:
            self.model[state] = update
        else:
            self.model[state] += update

    def state_dict(self):
        return self.model


class Actor(object):
    def __init__(self, num_actions, tie_break='next', tie_threshold=0.0):
        self.num_actions = num_actions
        self.tie_break = tie_break
        self.tie_threshold = tie_threshold
        self.action_count = 0
        self.tied = False

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


def R_update(state, reward, critic, lr):
    """Really simple TD learning"""

    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def E_update(state, value, critic, lr):
    """Bellman update"""
    update = lr * value
    critic.update_(state, update, replace=True)

    return critic


def R_homeostasis(reward, total_reward, set_point):
    """Update reward value assuming homeostatic value.
    
    Value based on Keramati and Gutkin, 2014.
    https://elifesciences.org/articles/04811
    """
    deviance_last = np.abs(set_point - total_reward)
    deviance = np.abs(set_point - (total_reward + reward))
    reward_value = deviance_last - deviance
    return reward_value


def run(env_name='BanditOneHot10-v0',
        num_episodes=1000,
        tie_break='next',
        tie_threshold=0.0,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None):
    """Bandit agent - argmax (E, R)"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    default_reward_value = 0
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value
    R_t = default_reward_value

    # -
    critic_R = Critic(env.observation_space.n,
                      default_value=default_reward_value)
    critic_E = Critic(env.observation_space.n,
                      default_value=default_info_value)
    actor_R = Actor(num_actions,
                    tie_break='first',
                    tie_threshold=tie_threshold)
    actor_E = Actor(num_actions,
                    tie_break=tie_break,
                    tie_threshold=tie_threshold)
    memories = [Count() for _ in range(num_actions)]
    all_actions = list(range(num_actions))

    # -
    num_best = 0
    total_R = 0.0
    total_E = 0.0
    total_regret = 0.0

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

        # Meta-greed policy selection
        if (E_t - tie_threshold) > R_t:
            critic = critic_E
            actor = actor_E
            policy = 0
        else:
            critic = critic_R
            actor = actor_R
            policy = 1

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regret = estimate_regret(all_actions, action, critic)

        # Pull a lever.
        state, R_t, _, _ = env.step(action)
        R_t = R_homeostasis(R_t, total_R, num_episodes)
        state = int(state[0])

        # Estimate E
        old = deepcopy(memories[action])
        memories[action].update(R_t)
        new = deepcopy(memories[action])
        E_t = kl(new, old, default_info_value)

        # Learning, both policies.
        critic_R = R_update(action, R_t, critic_R, lr_R)
        critic_E = E_update(action, E_t, critic_E, lr=1)

        # Log data
        writer.add_scalar("policy", policy, n)
        writer.add_scalar("state", state, n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_E", critic_E(action), n)
        writer.add_scalar("value_R", critic_R(action), n)

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
                  num_episodes=num_episodes,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold,
                  critic_E=critic_E.state_dict(),
                  critic_R=critic_R.state_dict(),
                  total_E=total_E,
                  total_R=total_R,
                  total_regret=total_regret,
                  env_name=env_name,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)