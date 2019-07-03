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


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def save_checkpoint(state, filename='checkpoint.pkl'):
    data = cloudpickle.dumps(state)
    with open(filename, 'wb') as fi:
        fi.write(data)


def load_checkpoint(filename='checkpoint.pkl'):
    with open(filename, 'rb') as fi:
        return cloudpickle.load(fi)


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        seed_value=42,
        lr_R=.1,
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
    best_action = env.env.best

    # -
    default_reward_value = 0  # Null R
    R_t = default_reward_value
    critic = Critic(
        env.observation_space.n, default_value=default_reward_value)

    # ------------------------------------------------------------------------
    # Play
    num_best = 0
    total_R = 0.0
    scores_R = []
    values_R = []
    actions = []
    p_bests = []
    regrets = []
    visited_states = set()
    states = list(range(num_actions))

    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])
        action = int(np.random.random_integers(0, num_actions))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regrets.append(estimate_regret(states, action, critic))

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        state = int(state[0])
        R_t = reward  # Notation consistency

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        visited_states.add(action)  # Action is state here
        actions.append(action)
        total_R += R_t
        scores_R.append(R_t)
        values_R.append(critic(action))
        p_bests.append(num_best / (n + 1))

        # -
        if debug:
            print(
                f">>> State {state}, Action {action}, Rt {R_t}, Epsilon {actor.epsilon}"
            )
            print(f">>> critic_R: {critic.state_dict()}")
        if progress:
            print(f">>> Episode {n}.")
        if progress or debug:
            print(f">>> Total R: {total_R}")

    # -
    episodes = list(range(num_episodes))
    result = dict(
        best=env.env.best,
        episodes=episodes,
        num_episodes=num_episodes,
        actions=actions,
        p_bests=p_bests,
        regrets=regrets,
        visited_states=visited_states,
        critic_R=critic.state_dict(),
        total_R=total_R,
        scores_R=scores_R,
        values_R=values_R)

    # Save models to disk when done?
    if save is not None:
        save_checkpoint(result, filename=save)

    if interactive:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)