import fire
import gym
import torch

from infomercial.util import Distribution
from infomercial.discrete.policy import greedy
from infomercial.discrete.policy import anti_greedy
from infomercial.discrete.value import information_value

from collections import OrderedDict


def create_Q():
    """Create a Q table"""
    return OrderedDict()


def Q_learn(state, reward, Q, lr):
    """Really simple Q learning"""
    if state in Q:
        Q[state] += (lr * (reward - Q[state]))
    else:
        Q[state] += (lr * reward)
    return Q


def run(env_name, num_episodes=10, lr=1, progress=True, debug=False):
    """Play some slots!"""

    # ------------------------------------------------------------------------
    # Init
    env = gym.make(env_name)

    Q_reward = create_Q()
    Q_relevance = create_Q()
    state_dist = Distribution()

    total_reward = 0.0
    total_relevance = 0.0

    # ------------------------------------------------------------------------
    # Play
    for n in range(num_episodes):
        state = env.reset()

        # Pick a policy. Choose an action.
        if total_reward > total_relevance:
            action, _ = greedy(list(Q_reward.values()))
        else:
            action, _ = anti_greedy(list(Q_relevance.values()))

        # Pull a lever.
        state, reward, _, _ = env.step(action)

        # Est. relevance
        prob1 = state_dist.values()
        state_dist.update(state)
        prob2 = state_dist.values()
        relevance = information_value(prob1, prob2)

        # Learn!
        Q_reward = Q_learn(state, reward, Q_reward, lr)
        Q_relevance = Q_learn(state, relevance, Q_relevance, lr)

        # Add to winnings
        total_reward += reward
        total_relevance += relevance

        if debug:
            print(f">>> Q_reward: {Q_reward}")
            print(f">>> Q_reward: {Q_relevance}")
        if progress:
            print(
                f">>> Episode {n}, Reward {total_reward}, Relevance {relevance}"
            )
    return total_reward, total_relevance, Q_reward, Q_relevance, state_dist


if __name__ == "__main__":
    fire.Fire(run)
