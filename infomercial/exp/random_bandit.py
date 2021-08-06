import os
import fire
import gym
import cloudpickle
import numpy as np

from noboard.csv import SummaryWriter

from scipy.stats import entropy
from infomercial.models import RandomActor
from infomercial.models import Critic
from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint


def Q_update(state, reward, critic, lr):
    """Really simple Q learning"""
    update = lr * (reward - critic(state))
    critic.update(state, update)

    return critic


def run(env_name='BanditOneHot2-v0',
        num_episodes=1,
        lr_R=.1,
        master_seed=42,
        write_to_disk=True,
        log_dir=None,
        output=False):
    """Bandit agent - random"""

    # --- Init ---
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # -
    env = gym.make(env_name)
    env.seed(master_seed)
    num_actions = env.action_space.n
    best_action = env.best

    # -
    default_reward_value = 0  # Null R
    R_t = default_reward_value
    critic = Critic(num_actions, default_value=default_reward_value)
    actor = RandomActor(num_actions, seed_value=master_seed)
    all_actions = list(range(num_actions))

    # ------------------------------------------------------------------------
    num_best = 0
    total_R = 0.0
    total_regret = 0.0

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

        # Critic learns
        critic = Q_update(action, R_t, critic, lr_R)

        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_R", critic(action), n)

        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)

    # -- Build the final result, and save or return it ---
    writer.close()
    result = dict(best=env.best,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  lr_R=lr_R,
                  critic=critic.state_dict(),
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