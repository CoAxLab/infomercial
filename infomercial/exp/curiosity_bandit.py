import fire
import gym

import numpy as np

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


def softmax(X, beta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Note
    ----

    Taken from: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html

    Parameters
    ----------
    X: Array. 
    beta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the beta parameter,
    y = y * float(beta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class SoftmaxActor(object):
    def __init__(self, num_actions, beta=1.0, tie_threshold=0.0, seed=None):
        self.prng = np.random.RandomState(seed)
        self.beta = beta
        self.tie_threshold = tie_threshold
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        self.action_count = 0

        # Undef for softmax. Set to False: API consistency.
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        values = np.asarray(values)
        values[values < self.tie_threshold] = 1e-16  # 'zer0'
        probs = softmax(values, self.beta)
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


def update_E(state, value, critic, lr):
    """Bellman update"""
    update = lr * value
    critic.update_(state, update, replace=True)

    return critic


# def E_oracle(actions, critic, memories, env, default_info_value):
#     """Find the best E, by one step rollout on the env"""
#     possible_values = []
#     for action in actions:
#         env_copy = deepcopy(env)
#         m_copy = deepcopy(memories)

#         state, _, _, _ = env_copy.step(action)

#         E, _ = estimate_E(action, state, m_copy, default_info_value)
#         possible_values.append(E)

#     return np.max(possible_values)


def estimate_E(action, state, memories, default_info_value):
    old = deepcopy(memories[action])
    memories[action].update(state)
    new = deepcopy(memories[action])

    return kl(new, old, default_info_value), memories


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        tie_break='next',
        tie_threshold=0.0,
        beta=None,
        seed_value=42,
        reward_mode=False,
        save=None,
        progress=False,
        debug=False,
        interactive=True):
    """Play some slots!"""

    # --- Init ---
    env = gym.make(env_name)
    env.seed(seed_value)

    num_actions = env.action_space.n
    best_action = env.best
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value

    # Agents and memories
    visited_arms = set()
    all_actions = list(range(num_actions))

    critic_E = Critic(num_actions, default_value=default_info_value)
    if beta is None:
        actor_E = DeterministicActor(num_actions,
                                     tie_break=tie_break,
                                     tie_threshold=tie_threshold)
    else:
        actor_E = SoftmaxActor(num_actions,
                               beta=beta,
                               tie_threshold=tie_threshold,
                               seed=seed_value)

    memories = [Count() for _ in range(num_actions)]

    # --- Init log ---
    num_best = 0
    total_E = 0.0
    scores_E = []
    values_E = []
    actions = []
    regrets = []
    p_bests = []
    ties = []
    policies = []

    # --- Main loop ---
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Each ep resets the env
        env.reset()

        # Choose a bandit arm
        values = list(critic_E.model.values())
        action = actor_E(values)
        if action in best_action:
            num_best += 1

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        visited_arms.add(action)
        if reward_mode:
            state = reward

        # Estimate E
        E_t, memories = estimate_E(action, state, memories, default_info_value)

        # Est regrets
        regrets.append(estimate_regret(all_actions, action, critic_E))

        # Learn
        critic_E = update_E(action, E_t, critic_E, lr=1)

        # Log data
        actions.append(action)
        total_E += E_t
        scores_E.append(E_t)
        values_E.append(critic_E(action) - tie_threshold)
        p_bests.append(num_best / (n + 1))

        if actor_E.tied:
            ties.append(1)
        else:
            ties.append(0)

        if debug:
            print(f">>> critic_E: {critic_E.state_dict()}")
        if progress:
            print(f">>> Episode {n}.")
        if progress or debug:
            print(f">>> Total E: {total_E}\n")

    # -- Build the result, and save or return it ---
    episodes = list(range(num_episodes))
    result = dict(best=best_action,
                  episodes=episodes,
                  policies=policies,
                  actions=actions,
                  p_bests=p_bests,
                  ties=ties,
                  critic_E=critic_E.state_dict(),
                  total_E=total_E,
                  scores_E=scores_E,
                  values_E=values_E,
                  regrets=regrets,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold)
    if save is not None:
        save_checkpoint(result, filename=save)
    if interactive:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)