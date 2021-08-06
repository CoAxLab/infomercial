import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from copy import deepcopy
from scipy.stats import entropy

from collections import OrderedDict

from infomercial.memory import CountMemory
from infomercial.models import Critic
from infomercial.models import SoftmaxActor

from infomercial.utils import estimate_regret
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update(state, update)

    return critic


def run(env_name='BanditOneHigh2-v0',
        num_episodes=1,
        temp=1.0,
        beta=1.0,
        lr_R=.1,
        mode='EB',
        master_seed=42,
        write_to_disk=True,
        load=None,
        log_dir=None,
        output=False):
    """Bandit agent - sample(R + beta count^(-1/2))"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))
    best_action = env.best

    default_reward_value = 0  # Null R
    R_t = default_reward_value

    # Agents and memories
    critic = Critic(num_actions, default_value=default_reward_value)
    actor = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    count = CountMemory()

    # Update with pre-loaded data. This will let you run
    # test experiments on pre-trained model and/or to
    # continue training.
    if load is not None:
        result = load_checkpoint(load)
        critic.load_state_dict(result['critic'])
        count.load_state_dict(result['count'])

    # -
    total_R = 0.0
    total_regret = 0.0
    num_best = 0

    # ------------------------------------------------------------------------
    for n in range(num_episodes):
        env.reset()

        # Choose an action; Choose a bandit
        action = actor(list(critic.model.values()))
        if action in best_action:
            num_best += 1

        # Est. regret and save it
        regret = estimate_regret(all_actions, action, critic)

        # Pull a lever.
        state, R_t, _, _ = env.step(action)

        # Apply count bonus
        count.update(action)

        if mode == "EB":
            count_bonus = count(action)**(-0.5)
        elif mode == "UCB":
            count_bonus = ((2 * np.log(n + 1)) / count(action))**(0.5)
        else:
            raise ValueError("mode must be EB or UCB")

        # Critic learns
        payout = R_t + (beta * count_bonus)
        critic = Q_update(action, payout, critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("bonus", count_bonus, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_R", critic(action), n)

        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

    # --- Build the final result, and save or return it ---
    writer.close()

    result = dict(best=env.best,
                  beta=beta,
                  temp=temp,
                  mode=mode,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  critic=critic.state_dict(),
                  count=count.state_dict(),
                  total_R=total_R,
                  total_regret=total_regret,
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