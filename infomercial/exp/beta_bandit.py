import fire
import gym
import cloudpickle
import numpy as np

from scipy.stats import entropy
from infomercial.memory import ConditionalCount
from infomercial.policy import greedy
from infomercial.utils import estimate_regret

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


def information_value(p_new, p_old, base=None):
    """Calculate information value."""
    if np.isclose(np.sum(p_old), 0.0):
        return 0.0  # Hack

    return entropy(p_old, qk=p_new, base=base)


def save_checkpoint(state, filename='checkpoint.pkl'):
    data = cloudpickle.dumps(state)
    with open(filename, 'wb') as fi:
        fi.write(data)


def load_checkpoint(filename='checkpoint.pkl'):
    with open(filename, 'rb') as fi:
        return cloudpickle.load(fi)


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

    # -
    default_reward_value = 0  # Null R
    default_info_value = entropy(np.ones(num_actions) /
                                 num_actions)  # Uniform p(a)

    critic = Critic(env.observation_space.n,
                    default_value=default_reward_value +
                    (beta * default_info_value))
    actor = Actor(num_actions,
                  tie_break=tie_break,
                  tie_threshold=tie_threshold)

    best_action = env.best

    # -
    memory = ConditionalCount()
    visited_states = set()
    states = list(range(num_actions))
    E_t = 0.0
    R_t = 0.0

    # ------------------------------------------------------------------------
    # Play
    total_R = 0.0
    total_E = 0.0
    num_best = 0
    scores_E = []
    scores_R = []
    values = []
    actions = []
    p_bests = []
    regrets = []
    ties = []
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

        # Build memory sampling lists, state:
        # r in (0,1); cond: bandit code
        cond_sample = list(visited_states) * 2
        state_sample = [0] * len(visited_states) + [1] * len(visited_states)

        # Update the memory and est. information value of the state
        p_old = memory.probs(state_sample, cond_sample)
        memory.update(reward, action)
        p_new = memory.probs(state_sample, cond_sample)

        info = information_value(p_new, p_old)
        E_t = info

        # -
        if debug:
            print(f">>> State {state}, Action {action}, Rt {R_t}, Et {E_t}")
            print(f">>> Cond sample: {cond_sample}")
            print(f">>> State sample: {state_sample}")
            print(f">>> p_old: {p_old}")
            print(f">>> p_new: {p_new}")
            print(f">>> E_t: {E_t}\n")

        # Critic learns
        critic = Q_update(action, R_t + (beta * E_t), critic, lr_R)

        # Log data
        actions.append(action)
        if actor.tied:
            ties.append(1)
        else:
            ties.append(0)
        total_R += R_t
        total_E += beta * E_t
        scores_E.append(beta * E_t)
        scores_R.append(R_t)
        values.append(critic(action))
        p_bests.append(num_best / (n + 1))

        # -
        if debug:
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
                  tie_break=tie_break,
                  tie_threshold=tie_threshold,
                  episodes=episodes,
                  actions=actions,
                  p_bests=p_bests,
                  regrets=regrets,
                  ties=ties,
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