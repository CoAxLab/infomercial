import fire
import gym
import cloudpickle
import numpy as np

from scipy.stats import entropy
from infomercial.memory import ConditionalCount
from infomercial.policy import greedy

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
        if self.prng.rand() < self.epsilon:
            action = self.prng.randint(0, self.num_actions, size=1)[0]
        else:
            action = np.argmax(values)

        return action


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


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        epsilon=0.1,
        epsilon_decay_tau=0,
        lr=.1,
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
    R_t = default_reward_value
    critic = Critic(
        env.observation_space.n, default_value=default_reward_value)
    actor = Actor(num_actions, epsilon=epsilon, decay_tau=epsilon_decay_tau)

    # ------------------------------------------------------------------------
    # Play
    total_R = 0.0
    scores_R = []
    values_R = []
    actions = []
    epsilons = []
    visited_states = set()
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

        # Choose an action; Choose a bandit
        values = list(critic.model.values())
        action = actor(values)

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        state = int(state[0])
        R_t = reward  # Notation consistency

        # Critic learns
        critic = Q_update(action, R_t, critic, lr)

        # Decay ep. noise?
        if epsilon_decay_tau > 0:
            actor.decay_epsilon()

        # Log data
        visited_states.add(action)  # Action is state here
        actions.append(action)
        total_R += R_t
        scores_R.append(R_t)
        values_R.append(critic(action))
        epsilons.append(actor.epsilon)

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
        actions=actions,
        epsilons=epsilons,
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