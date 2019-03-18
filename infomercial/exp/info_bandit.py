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
        self.model[state] = self.default_value
        return self.model[state]

    def update_(self, state, update):
        self.model[state] += update

    def state_dict(self):
        return self.model


class Actor(object):
    def __init__(self, num_actions, tie_break='first'):
        self.num_actions = num_actions
        self.tie_break = tie_break

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        if self.tie_break == 'first':
            return np.argmax(values)
        else:
            raise ValueError("tie_break must be 'first'")


def information_value(p_new, p_old, base=None):
    """Calculate information value."""
    return entropy(p_old, qk=p_new, base=base)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    data = cloudpickle.dumps(state)
    with open(filename, 'w') as fi:
        fi.write(data)


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update_(state, update)

    return critic


def run(env_name='BanditTwoArmedDeterministicFixed-v0',
        num_episodes=1,
        policy_mode='meta',
        tie_break='first',
        default_value=0.0,
        lr=1,
        save=None,
        progress=True,
        debug=False):
    """Play some slots!"""

    # ------------------------------------------------------------------------
    # Init
    env = gym.make(env_name)

    critic_R = Critic(env.observation_space.n, default_value=default_value)
    critic_E = Critic(env.observation_space.n, default_value=default_value)
    actor = Actor(env.action_space.n, tie_break=tie_break)

    memory = ConditionalCount()

    if policy_mode == 'meta':
        E_t = 0.0
        R_t = 0.0

    # TODO: add R or E only; use an np.inf assignment?
    else:
        raise ValueError("policy mode must be 'meta'")

    total_R = 0.0
    total_E = 0.0

    scores_E = []
    scores_R = []

    # ------------------------------------------------------------------------
    # Play
    for n in range(num_episodes):
        # Every play is also an ep for bandit tasks.
        state = env.reset()

        # Pick a critic, which will in turn choose the action policy
        if E_t > R_t:
            critic = critic_E
        else:
            critic = critic_R

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        R_t = reward  # Notation consistency

        # Update the memory and est. information value of the state
        probs_old = memory.probs()
        memory.update(reward, state)
        probs_new = memory.probs()
        info = information_value(probs_new, probs_old)
        E_t = info

        if debug:
            print(
                f">>> Episode {n}: State {state}, Action {action}, Rt {R_t}, Et {E_t}"
            )

        # Critic learns
        critic_R = Q_update(state, R_t, critic_R, lr)
        critic_E = Q_update(state, E_t, critic_E, lr)

        # Add to winnings
        total_R += R_t
        total_E += E_t
        scores_E.append(E_t)
        scores_R.append(R_t)

        if debug:
            print(f">>> critic_R: {critic_R}")
            print(f">>> critic_E: {critic_E}")

        if progress or debug:
            print(f">>> Episode {n}, Reward {total_R}, Relevance {total_E}")

    # Save models to disk
    if save is not None:
        save_checkpoint(
            dict(
                critic_E=critic_E.state_dict(),
                critic_R=critic_R.state_dict(),
                total_E=total_E,
                total_R=total_R,
                scores_E=scores_E,
                scores_R=scores_R),
            filename=save)

    return list(range(num_episodes)), scores_E, scores_R


if __name__ == "__main__":
    fire.Fire(run)