import fire
import gym
import cloudpickle

from scipy.stats import entropy
from infomercial.memory import ConditionalCount
from infomercial.policy import greedy

from collections import OrderedDict


def information_value(p_new, p_old, base=None):
    """Calculate information value."""
    return entropy(p_old, qk=p_new, base=base)


def create_critic():
    """Create a Q table"""
    return OrderedDict()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    data = cloudpickle.dumps(state)
    with open(filename, 'w') as fi:
        fi.write(data)


def Q_update(state, reward, Q, lr):
    """Really simple Q learning"""
    if state in Q:
        Q[state] += (lr * (reward - Q[state]))
    else:
        Q[state] += (lr * reward)
    return Q


def run(env_name,
        num_episodes,
        policy_mode='meta',
        lr=1,
        save=None,
        progress=True,
        debug=False):
    """Play some slots!"""

    # ------------------------------------------------------------------------
    # Init
    env = gym.make(env_name)

    critic_R = create_critic()
    critic_E = create_critic()
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

        # Choose an action
        action, _ = greedy(list(critic.values()))

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
                critic_E=critic_E.items(),
                critic_R=critic_R.items(),
                total_E=total_E,
                total_R=total_R,
                scores_E=scores_E,
                scores_R=scores_R),
            filename=save)

    return list(range(num_episodes)), scores_E, scores_R


if __name__ == "__main__":
    fire.Fire(run)