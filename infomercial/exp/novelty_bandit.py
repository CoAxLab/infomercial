import fire
import gym
import cloudpickle
import numpy as np

from scipy.stats import entropy
# from infomercial.memory import ConditionalCount
from infomercial.policy import greedy
from infomercial.utils import estimate_regret

from collections import OrderedDict

from infomercial.exp.epsilon_bandit import Q_update
from infomercial.exp.epsilon_bandit import Actor
from infomercial.exp.epsilon_bandit import Critic
from infomercial.exp.epsilon_bandit import save_checkpoint
from infomercial.exp.epsilon_bandit import load_checkpoint


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        epsilon=0.1,
        epsilon_decay_tau=0,
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
    novelty_bonus = np.ones(num_actions)

    # -
    default_reward_value = 0  # Null R
    R_t = default_reward_value
    critic = Critic(env.observation_space.n,
                    default_value=default_reward_value)
    actor = Actor(num_actions,
                  epsilon=epsilon,
                  decay_tau=epsilon_decay_tau,
                  seed_value=seed_value)

    # ------------------------------------------------------------------------
    # Play
    num_best = 0
    total_R = 0.0
    scores_R = []
    novelty_R = []
    values_R = []
    actions = []
    p_bests = []
    regrets = []
    epsilons = []
    visited_states = set()
    states = list(range(num_actions))

    for n in range(num_episodes):
        if debug:
            print(f"\n>>> Episode {n}")

        # Every play is also an ep for bandit tasks.
        # Thus this reset() call
        state = int(env.reset()[0])

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

        # Add novelty bonus, then zero it.
        bonus = novelty_bonus[int(action)]
        reward += bonus
        novelty_bonus[int(action)] = 0

        # Notation consistency
        R_t = reward

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Decay ep. noise?
        if epsilon_decay_tau > 0:
            actor.decay_epsilon()

        # Log data
        visited_states.add(action)  # Action is state here
        actions.append(action)
        total_R += R_t
        scores_R.append(R_t)
        novelty_R.append(bonus)
        values_R.append(critic(action))
        epsilons.append(actor.epsilon)
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
    result = dict(best=env.best,
                  episodes=episodes,
                  num_episodes=num_episodes,
                  lr_R=lr_R,
                  actions=actions,
                  p_bests=p_bests,
                  regrets=regrets,
                  epsilons=epsilons,
                  visited_states=visited_states,
                  critic_R=critic.state_dict(),
                  total_R=total_R,
                  scores_R=scores_R,
                  novelty_R=novelty_R,
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