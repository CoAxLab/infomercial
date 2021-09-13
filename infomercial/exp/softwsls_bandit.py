import os
import fire
import gym

import numpy as np
from scipy.special import softmax

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy
from collections import OrderedDict

from infomercial.distance import kl
from infomercial.memory import DiscreteDistribution
from infomercial.models import Critic
from infomercial.models import SoftmaxActor

from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint


def R_update(state, reward, critic, lr):
    """Really simple TD learning"""

    update = lr * (reward - critic(state))
    critic.update(state, update)

    return critic


def E_update(state, value, critic, lr):
    """Bellman update"""
    update = lr * value
    critic.replace(state, update)

    return critic


def R_homeostasis(reward, total_reward, set_point):
    """Update reward value assuming homeostatic value.
    
    Value based on Keramati and Gutkin, 2014.
    https://elifesciences.org/articles/04811
    """
    deviance_last = np.abs(set_point - total_reward)
    deviance = np.abs(set_point - (total_reward + reward))
    reward_value = deviance_last - deviance
    return reward_value


def run(env_name='BanditOneHot10-v0',
        num_episodes=1000,
        temp=1.0,
        tie_threshold=0.0,
        tie_break=None,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None,
        output=True):
    """Bandit agent - softmax (E, R)"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))
    best_action = env.best

    default_reward_value = 0
    default_info_value = entropy(np.ones(num_actions) / num_actions)
    E_t = default_info_value
    R_t = default_reward_value

    # --- Agents and memories ---
    critic_R = Critic(num_actions, default_value=default_reward_value)
    critic_E = Critic(num_actions, default_value=default_info_value)
    actor_R = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    actor_E = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    memories = [DiscreteDistribution() for _ in range(num_actions)]

    # -
    num_best = 0
    total_R = 0.0
    total_E = 0.0
    total_regret = 0.0

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        env.reset()

        # Meta-greed policy selection
        if (E_t - tie_threshold) > R_t:
            critic = critic_E
            actor = actor_E
            policy = 0
        else:
            critic = critic_R
            actor = actor_R
            policy = 1

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regret = estimate_regret(all_actions, action, critic)

        # Pull a lever.
        state, R_t, _, _ = env.step(action)
        R_t = R_homeostasis(R_t, total_R, num_episodes)

        # Estimate E
        old = deepcopy(memories[action])
        memories[action].update((int(state), int(R_t)))
        new = deepcopy(memories[action])
        E_t = kl(new, old, default_info_value)

        # Learning, both policies.
        critic_R = R_update(action, R_t, critic_R, lr_R)
        critic_E = E_update(action, E_t, critic_E, lr=1)

        # Log data
        writer.add_scalar("policy", policy, n)
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_E", critic_E(action), n)
        writer.add_scalar("value_R", critic_R(action), n)

        total_E += E_t
        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_E", total_E, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

    # -- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=env.best,
                  num_episodes=num_episodes,
                  temp=temp,
                  tie_threshold=tie_threshold,
                  critic_E=critic_E.state_dict(),
                  critic_R=critic_R.state_dict(),
                  total_E=total_E,
                  total_R=total_R,
                  total_regret=total_regret,
                  env_name=env_name,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    if output:
        return result
    else:
        return None


if __name__ == "__main__":
    fire.Fire(run)