import fire
import gym
import cloudpickle
import numpy as np

from copy import deepcopy
from scipy.stats import entropy

from collections import OrderedDict

from infomercial.memory import Count
from infomercial.utils import estimate_regret
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint
from infomercial.distance import kl


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


class SoftmaxActor(object):
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


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditOneHigh2-v0',
        num_episodes=1,
        beta=1.0,
        lr_R=.1,
        temp=1,
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

    default_reward_value = 0  # Null R
    default_info_value = entropy(np.ones(num_actions) /
                                 num_actions)  # Uniform p(a)
    E_t = default_info_value
    R_t = default_reward_value

    visited_states = set()
    states = list(range(num_actions))

    # Agents and memories
    critic = Critic(env.observation_space.n,
                    default_value=default_reward_value +
                    (beta * default_info_value))
    actor = SoftmaxActor(num_actions, temp=temp, seed_value=seed_value)
    memories = [Count() for _ in range(num_actions)]

    # Init log
    total_R = 0.0
    total_E = 0.0
    num_best = 0
    scores_E = []
    scores_R = []
    values = []
    actions = []
    p_bests = []
    regrets = []

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
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

        # Critic learns
        critic = Q_update(action, R_t + (beta * E_t), critic, lr_R)

        # Log data
        actions.append(action)
        total_R += R_t
        total_E += beta * E_t
        scores_E.append(beta * E_t)
        scores_R.append(R_t)
        values.append(critic(action))
        p_bests.append(num_best / (n + 1))
        if debug:
            print(f">>> State {state}, Action {action}, Rt {R_t}, Et {E_t}")
            print(f">>> E_t: {E_t}\n")
            print(f">>> critic: {critic.state_dict()}")
        if progress:
            print(f">>> Episode {n}.")
        if progress or debug:
            print(f">>> Total R: {total_R}; Total E: {total_E}\n")

    # -
    # Save models to disk when done?
    episodes = list(range(num_episodes))
    result = dict(best=env.best,
                  lr_R=lr_R,
                  beta=beta,
                  temp=temp,
                  episodes=episodes,
                  actions=actions,
                  p_bests=p_bests,
                  regrets=regrets,
                  critic=critic.state_dict(),
                  total_E=total_E,
                  total_R=total_R,
                  scores_E=scores_E,
                  scores_R=scores_R,
                  values_R=values)

    if save is not None:
        save_checkpoint(result, filename=save)

    # -
    # Don't return anything if run from the CL
    if interactive:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)