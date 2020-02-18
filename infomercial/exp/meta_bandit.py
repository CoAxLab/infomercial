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


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        tie_break='next',
        tie_threshold=0.0,
        lr_R=.1,
        seed_value=42,
        save=None,
        progress=False,
        debug=False,
        interactive=True):
    """Play some slots!"""

    # ------------------------------------------------------------------------
    # Init
    env = gym.make(env_name)
    env.seed(seed_value)
    num_actions = env.action_space.n
    best_action = env.best

    default_reward_value = 0
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value
    R_t = default_reward_value

    visited_states = set()
    states = list(range(num_actions))

    # -
    # Agents and memories
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

    # -
    # Init log
    num_best = 0
    total_R = 0.0
    total_E = 0.0
    scores_E = []
    scores_R = []
    values_E = []
    values_R = []
    actions = []
    regrets = []
    p_bests = []
    ties = []
    policies = []

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

        # Use the the meta-greedy policy to
        # pick an actor, critic pair.
        if (E_t - tie_threshold) > R_t:
            critic = critic_E
            actor = actor_E
            policies.append(0)
            if debug: print(f">>> E in control!")
        else:
            critic = critic_R
            actor = actor_R
            policies.append(1)
            if debug: print(f">>> R in control!")

        # Choose an action; Choose a bandit
        values = list(critic.model.values())
        action = actor(values)
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regrets.append(estimate_regret(states, action, critic))

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        state = int(state[0])
        R_t = reward  # Notation consistency
        visited_states.add(action)  # Action is state here

        # Estimate E
        old = deepcopy(memories[action])
        memories[action].update(reward)
        new = deepcopy(memories[action])
        E_t = kl(new, old, default_info_value)

        # Learning, both policies.
        critic_R = R_update(action, R_t, critic_R, lr_R)
        critic_E = E_update(action, E_t, critic_E, lr=1)

        # Log data
        actions.append(action)
        total_R += R_t
        total_E += E_t
        scores_E.append(E_t)
        scores_R.append(R_t)
        values_E.append(critic_E(action))
        values_R.append(critic_R(action))
        p_bests.append(num_best / (n + 1))
        if actor.tied:
            ties.append(1)
        else:
            ties.append(0)
        if debug:
            print(f">>> State {state}, Action {action}, Rt {R_t}, Et {E_t}")
            print(f">>> critic_R: {critic_R.state_dict()}")
            print(f">>> critic_E: {critic_E.state_dict()}")
        if progress:
            print(f">>> Episode {n}.")
        if progress or debug:
            print(f">>> Total R: {total_R}; Total E: {total_E}\n")

    # -
    episodes = list(range(num_episodes))
    result = dict(best=best_action,
                  episodes=episodes,
                  policies=policies,
                  actions=actions,
                  p_bests=p_bests,
                  ties=ties,
                  critic_E=critic_E.state_dict(),
                  critic_R=critic_R.state_dict(),
                  total_E=total_E,
                  total_R=total_R,
                  total_E_R=total_E + total_R,
                  scores_E=scores_E,
                  scores_R=scores_R,
                  values_E=values_E,
                  values_R=values_R,
                  regrets=regrets,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold,
                  lr_R=lr_R)

    # Save models to disk when done?
    if save is not None:
        save_checkpoint(result, filename=save)

    if interactive:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)