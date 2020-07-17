import fire
import gym
import os

import numpy as np
from scipy.special import softmax

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
    def __init__(self,
                 num_inputs,
                 default_value,
                 default_noise_scale=0,
                 seed_value=None):

        self.num_inputs = num_inputs
        self.default_value = default_value
        self.default_noise_scale = default_noise_scale

        # Init
        self.model = OrderedDict()
        for n in range(self.num_inputs):
            # Def E0. Add noise? None by default.
            delta = 0.0

            if self.default_noise_scale > 0:
                prng = np.random.RandomState(seed_value)
                delta = prng.normal(0,
                                    scale=default_value *
                                    self.default_noise_scale)

            # Set E0
            self.model[n] = self.default_value + delta

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        return self.model[state]

    def update_(self, state, update):
        self.model[state] = update

    def state_dict(self):
        return self.model


class ThresholdActor(object):
    def __init__(self, num_actions, tie_threshold=0.0, seed=None):
        self.prng = np.random.RandomState(seed)
        self.tie_threshold = tie_threshold
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        self.action_count = 0
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Threshold
        values = np.asarray(values) - self.tie_threshold

        # Pick from postive values, while there still are positive values.
        mask = values > 0
        if np.sum(mask) > 0:
            filtered = [a for (a, m) in zip(self.actions, mask) if m]
            action = self.prng.choice(filtered)
        else:
            self.tied = True
            action = None

        return action


class RandomActor(object):
    def __init__(self, num_actions, tie_threshold=0.0, seed=None):
        self.prng = np.random.RandomState(seed)
        self.tie_threshold = tie_threshold
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        # Undef for softmax. Set to False: API consistency.
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        """Values are a dummy var. Pick at random"""

        # Threshold
        values = np.asarray(values) - self.tie_threshold

        # Move the the default policy?
        if np.sum(values < 0) == len(values):
            return None

        # Make a random choice
        action = self.prng.choice(self.actions)
        return action


class SoftmaxActor(object):
    def __init__(self, num_actions, beta=1.0, tie_threshold=0.0, seed=None):
        self.prng = np.random.RandomState(seed)
        self.beta = beta
        self.tie_threshold = tie_threshold
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))

        # Undef for softmax. Set to False: API consistency.
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Threshold
        values = np.asarray(values) - self.tie_threshold

        # Move the the default policy?
        if np.sum(values < 0) == len(values):
            return None

        # Make a soft choice
        probs = softmax(values * self.beta)
        action = self.prng.choice(self.actions, p=probs)

        return action


class DeterministicActor(object):
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
        t_values = [max(0, v) for v in values]

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
        # Threshold
        values = np.asarray(values) - self.tie_threshold

        # Move the the default policy?
        if np.sum(values < 0) == len(values):
            return None

        # Other wise go explore:
        # Pick the best as the base case, ....
        action = np.argmax(values)

        # then check for ties.
        if self.tie_break == 'first':
            pass
        elif self.tie_break == 'next':
            self.tied = self._is_tied(values)
            if self.tied:
                self.action_count += 1
                action = self.action_count % self.num_actions
        else:
            raise ValueError("tie_break must be 'first' or 'next'")

        return action


def update_E(state, E_t, critic, lr):
    """Bellman update"""
    V = critic(state)
    update = V + (lr * (E_t - V))
    critic.update_(state, update)

    return critic


def run(env_name='InfoBlueYellow4b-v0',
        num_episodes=1,
        lr_E=1.0,
        actor='DeterministicActor',
        initial_count=1,
        initial_bins=None,
        initial_noise=0.0,
        seed_value=42,
        reward_mode=False,
        log_dir=None,
        **actor_kwargs):
    """Play some slots!"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make(env_name)
    env.seed(seed_value)
    num_actions = env.action_space.n
    best_action = env.best
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value

    # --- Agents and memories ---
    # Critic
    all_actions = list(range(num_actions))
    critic_E = Critic(num_actions,
                      default_value=default_info_value,
                      default_noise_scale=initial_noise,
                      seed_value=seed_value)

    # Actor
    if actor == "DeterministicActor":
        actor_E = DeterministicActor(num_actions, **actor_kwargs)
    elif actor == "SoftmaxActor":
        actor_E = SoftmaxActor(num_actions, **actor_kwargs, seed=seed_value)
    elif actor == "RandomActor":
        actor_E = RandomActor(num_actions, **actor_kwargs, seed=seed_value)
    elif actor == "ThresholdActor":
        actor_E = ThresholdActor(num_actions, **actor_kwargs, seed=seed_value)
    else:
        raise ValueError("actor was not a valid choice")

    # Memory
    memories = [
        Count(intial_bins=initial_bins, initial_count=initial_count)
        for _ in range(num_actions)
    ]

    # --- Init log ---
    num_best = 0
    total_E = 0.0
    total_regret = 0.0

    # --- Main loop ---
    for n in range(num_episodes):
        # Each ep resets the env
        env.reset()

        # Choose a bandit arm
        values = list(critic_E.model.values())
        action = actor_E(values)
        if action is None:
            break
        regret = estimate_regret(all_actions, action, critic_E)

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        if reward_mode:
            state = reward

        # Estimate E, save regret
        old = deepcopy(memories[action])
        memories[action].update(state)
        new = deepcopy(memories[action])
        E_t = kl(new, old, default_info_value)
        # print(f"{action} - {E_t} - {old.values()}, {new.values()}")

        # --- Learn ---
        critic_E = update_E(action, E_t, critic_E, lr=lr_E)

        # --- Log data ---
        num_stop = n
        writer.add_scalar("state", state, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("value_E", critic_E(action), n)

        total_E += E_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_E", total_E, n)

        if action in best_action:
            num_best += 1
        writer.add_scalar("p_bests", num_best / (n + 1), n)
        writer.add_scalar("action", action, n)
        tie = 0
        if actor_E.tied:
            tie = 1
        writer.add_scalar("ties", tie, n)

    # -- Build the final result, and save or return it ---

    result = dict(best=best_action,
                  critic_E=critic_E.state_dict(),
                  total_E=total_E,
                  total_regret=total_regret,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  lr_E=lr_E,
                  seed_value=seed_value,
                  initial_bins=initial_bins,
                  initial_count=initial_count,
                  actor=actor,
                  actor_kwargs=actor_kwargs,
                  num_stop=num_stop + 1)

    # Save the result and flush, and close the writer
    save_checkpoint(result,
                    filename=os.path.join(writer.log_dir, "result.pkl"))
    writer.close()

    # -
    return result


if __name__ == "__main__":
    fire.Fire(run)