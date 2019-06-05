import fire
import gym
import cloudpickle
import numpy as np

from scipy.stats import entropy
from infomercial.memory import ConditionalCount
from infomercial.policy import greedy

from collections import OrderedDict


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
    best_action = env.env.best
    default_reward_value = 0  # Null R
    R_t = default_reward_value

    # ------------------------------------------------------------------------
    # Play
    num_best = 0
    total_R = 0.0
    scores_R = []
    actions = []
    p_bests = []
    visited_states = set()
    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])
        action = int(np.random.random_integers(0, num_actions))
        if action == best_action:
            num_best += 1

        # Pull a lever.
        state, reward, _, _ = env.step(action)
        state = int(state[0])
        R_t = reward  # Notation consistency

        # Log data
        visited_states.add(action)  # Action is state here
        actions.append(action)
        total_R += R_t
        scores_R.append(R_t)
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
        actions=actions,
        p_bests=p_bests,
        visited_states=visited_states,
        total_R=total_R,
        scores_R=scores_R)

    # Save models to disk when done?
    if save is not None:
        save_checkpoint(result, filename=save)

    if interactive:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)