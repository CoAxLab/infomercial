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
        v0 = t_values[0]
        for v in t_values[1:]:
            if np.isclose(v0, v):
                continue
            else:
                return False

        # Only get here if all values were the same.
        return True

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
            if self._is_tied(values):
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
        tie_break='next',
        tie_threshold=0.0,
        default_value=0.0,
        lr=.1,
        save=None,
        progress=False,
        debug=False):
    """Play some slots!"""

    # ------------------------------------------------------------------------
    # Init
    env = gym.make(env_name)

    # -
    critic_R = Critic(env.observation_space.n, default_value=default_value)
    critic_E = Critic(env.observation_space.n, default_value=default_value)

    num_actions = env.action_space.n
    actor_R = Actor(
        num_actions, tie_break=tie_break, tie_threshold=tie_threshold)
    actor_E = Actor(
        num_actions, tie_break=tie_break, tie_threshold=tie_threshold)

    # -
    memory = ConditionalCount()
    visited_states = set()

    # -
    if policy_mode == 'meta':
        E_t = 0.0
        R_t = 0.0
    # TODO: add R or E only; use an np.inf assignment?
    else:
        raise ValueError("policy mode must be 'meta'")

    # ------------------------------------------------------------------------
    # Play
    total_R = 0.0
    total_E = 0.0
    scores_E = []
    scores_R = []
    actions = []
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")
        # Every play is also an ep for bandit tasks.
        state = int(env.reset()[0])

        # Pick an actor, critic pair
        if E_t > R_t:
            critic = critic_E
            actor = actor_E
            if debug:
                print(f">>> E in control!")
        else:
            critic = critic_R
            actor = actor_R
            if debug:
                print(f">>> R in control!")

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        actions.append(action)

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        R_t = reward  # Notation consistency
        state = int(state[0])
        visited_states.add(action)

        # Build memory sampling lists, state: r in (0,1); cond: bandit code
        cond_sample = list(visited_states) * 2
        state_sample = [0] * len(visited_states) + [1] * len(visited_states)

        # Update the memory and est. information value of the state
        p_old = memory.probs(state_sample, cond_sample)
        memory.update(reward, action)
        p_new = memory.probs(state_sample, cond_sample)
        info = information_value(p_new, p_old)
        E_t = info

        if debug:
            print(f">>> State {state}, Action {action}, Rt {R_t}, Et {E_t}")
            print(f">>> Cond sample: {cond_sample}")
            print(f">>> State sample: {state_sample}")
            print(f">>> p_old: {p_old}")
            print(f"    p_new: {p_new}")

        # Critic learns

        critic_R = Q_update(action, R_t, critic_R, lr)
        critic_E = Q_update(action, E_t, critic_E, lr)

        # Add to winnings
        total_R += R_t
        total_E += E_t
        scores_E.append(critic_E(action))
        scores_R.append(critic_R(action))

        if debug:
            print(f">>> critic_R: {critic_R.state_dict()}")
            print(f">>> critic_E: {critic_E.state_dict()}")

        if progress:
            print(f">>> Episode {n}.")
        if progress or debug:
            print(f">>> Total R: {total_R}; Total E: {total_E}\n")

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

    return list(range(num_episodes)), actions, scores_E, scores_R


if __name__ == "__main__":
    fire.Fire(run)