import fire
import gym
import os

import numpy as np

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy
from collections import OrderedDict

from infomercial.distance import kl
from infomercial.memory import DiscreteDistribution
from infomercial.models import NoisyCritic
from infomercial.models import DeterministicActor
from infomercial.models import SoftmaxActor
from infomercial.models import ThresholdActor
from infomercial.models import RandomActor

from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint


def updateE(state, E_t, critic, lr):
    """Bellman update"""
    V = critic(state)
    update = V + (lr * (E_t - V))
    critic.update(state, update)

    return critic


def run(env_name='InfoBlueYellow4b-v0',
        num_episodes=1,
        lr_E=1.0,
        actor='DeterministicActor',
        initial_count=1,
        initial_bins=None,
        initial_noise=0.0,
        master_seed=None,
        env_seed=None,
        actor_seed=None,
        critic_seed=None,
        reward_mode=False,
        log_dir=None,
        write_to_disk=True,
        **actor_kwargs):
    """Play some slots!"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    if master_seed is not None:
        env_seed = master_seed
        critic_seed = master_seed
        actor_seed = master_seed

    # -
    env = gym.make(env_name)
    env.seed(env_seed)
    num_actions = env.action_space.n
    best_action = env.best
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value

    # --- Agents and memories ---
    # Critic
    all_actions = list(range(num_actions))
    critic_E = NoisyCritic(num_actions,
                           default_value=default_info_value,
                           default_noise_scale=initial_noise,
                           seed_value=critic_seed)

    # Actor
    if actor == "DeterministicActor":
        actor_E = DeterministicActor(num_actions, **actor_kwargs)
    elif actor == "SoftmaxActor":
        actor_E = SoftmaxActor(num_actions,
                               **actor_kwargs,
                               seed_value=actor_seed)
    elif actor == "RandomActor":
        actor_E = RandomActor(num_actions,
                              **actor_kwargs,
                              seed_value=actor_seed)
    elif actor == "ThresholdActor":
        actor_E = ThresholdActor(num_actions,
                                 **actor_kwargs,
                                 seed_value=actor_seed)
    else:
        raise ValueError("actor was not a valid choice")

    # Memory
    memories = [
        DiscreteDistribution(intial_bins=initial_bins,
                             initial_count=initial_count)
        for _ in range(num_actions)
    ]

    # --- Init log ---
    num_best = 0
    total_E = 0.0
    total_regret = 0.0

    # --- Main loop ---
    for n in range(num_episodes):
        # Each ep resets the env
        env.reset()

        # Choose a bandit arm
        values = list(critic_E.model.values())
        action = actor_E(values)
        if action is None:
            break
        regret = estimate_regret(all_actions, action, critic_E)

        # Pull a lever.
        state, reward, _, _ = env.step(action)

        # Estimate E, save regret
        old = deepcopy(memories[action])
        memories[action].update((int(state), int(reward)))
        new = deepcopy(memories[action])
        E_t = kl(new, old, critic_E.inital_values[action])

        # --- Learn ---
        critic_E = updateE(action, E_t, critic_E, lr=lr_E)

        # --- Log data ---
        num_stop = n
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("value_E", critic_E(action), n)

        total_E += E_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_E", total_E, n)

        if action in best_action:
            num_best += 1
        writer.add_scalar("p_bests", num_best / (n + 1), n)
        writer.add_scalar("action", action, n)
        tie = 0
        if actor_E.tied:
            tie = 1
        writer.add_scalar("ties", tie, n)

    # -- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=best_action,
                  critic_E=critic_E.state_dict(),
                  intial_E=list(critic_E.inital_values.values()),
                  total_E=total_E,
                  total_regret=total_regret,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  lr_E=lr_E,
                  master_seed=master_seed,
                  actor_seed=actor_seed,
                  critic_seed=critic_seed,
                  env_seed=env_seed,
                  initial_bins=initial_bins,
                  initial_count=initial_count,
                  actor=actor,
                  memories=[m.state_dict() for m in memories],
                  actor_kwargs=actor_kwargs,
                  num_stop=num_stop + 1)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    return result


if __name__ == "__main__":
    fire.Fire(run)