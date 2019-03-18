import fire
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

from infomercial import models
from infomercial.models import Memory
from infomercial.util import save_checkpoint
from infomercial.util import select_action

EPS = np.finfo(np.float32).eps.item()


def update(policy, memory, optimizer, batch_size, z_score=True, gamma=1.0):
    transitions = memory.sample(batch_size)
    _, _, log_probs, _, rewards = zip(*transitions)
    log_probs = list(log_probs)
    rewards = list(rewards)

    loss = []
    # Re-weight rewards with gamma. (Mem. eff. if ugly.)
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)

    # Norm rewards
    if z_score:
        rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

    # Log prob loss!
    for log_prob, reward in zip(log_probs, rewards):
        loss.append(-log_prob * reward)

    # Backprop teach us everything.
    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()

    return policy, loss


def train_model(env_name='BanditTwoArmedDeterministicFixed',
                num_episodes=100,
                batch_size=48,
                memory_size=1000,
                lr=0.001,
                learn=True,
                save=None,
                progress=True,
                debug=False,
                log_interval=1,
                render=False,
                seed=349,
                gamma=1.0,
                z_score=True,
                action_mode='Categorical',
                model_name='LinearCategorical',
                **model_hyperparameters):
    """Learn with REINFORCE!"""

    # ------------------------------------------------------------------------
    # Sanity
    if num_episodes < batch_size:
        raise ValueError("num_episodes must be >= batch_size")
    if memory_size < batch_size:
        raise ValueError("memory_size must be >= batch_size")
    if lr < 0:
        raise ValueError("lr must be > 0.0")
    if gamma < 0:
        raise ValueError("gamma must be > 0.0")
    if log_interval < 0:
        raise ValueError("log_interval must be > 0.0")

    # ------------------------------------------------------------------------
    # Setup the world
    prng = np.random.RandomState(seed)
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)

    # Note its size
    try:
        num_inputs = env.observation_space.shape[0]
    except IndexError:
        num_inputs = 1
    try:
        num_actions = env.action_space.shape[0]
    except IndexError:
        num_actions = 1

    # ------------------------------------------------------------------------
    # Model init
    Model = getattr(models, model_name)
    policy = Model(**model_hyperparameters)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    memory = Memory(memory_size)

    # ------------------------------------------------------------------------
    # Run some games!
    total_reward = 0.0
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state)

        if progress and (episode % log_interval) == 0:
            print(f"--- Episode {episode} ---")
        if debug and (episode % log_interval) == 0:
            print(f"--- Episode {episode} ---")
            print(f">>> Initial state {state}")

        done = False
        rewards = []
        while not done:  # Don't infinite loop while learning
            # Act!
            policy, action, log_prob = select_action(
                policy, state, mode=action_mode)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state)

            # Remeber this transition set
            memory.push(state, action, log_prob, next_state, reward)
            rewards.append(reward)

            # Shift state
            state = next_state

            if debug and (episode % log_interval) == 0:
                print(f">>> Action {action}")
                print(f">>> p(action) {np.exp(log_prob.detach())}")
                print(f">>> Next state {state}")
                print(f">>> Reward {reward}")
            if render:
                env.render()

        avg_r = np.mean(rewards)
        total_reward += avg_r

        loss = None
        if learn and (len(memory) >= batch_size):
            policy, loss = update(
                policy,
                memory,
                optimizer,
                batch_size=batch_size,
                z_score=z_score,
                gamma=gamma)

            # REINFORCE doesn't share memory, classically
            memory.reset()

            if ((episode % log_interval) == 0 and (progress or debug)):
                print(">>> UPDATING the policy!")
                print(f">>> Loss {loss}")

        # --------------------------------------------------------------------
        if (save is not None) and (episode % log_interval) == 0:
            save_checkpoint(
                {
                    'actor': policy.state_dict(),
                    'episode_score': avg_r,
                    'total_score': total_reward,
                    "loss": loss
                },
                filename=save + "_ep_{}.pytorch.tar".format(episode))

        # --------------------------------------------------------------------
        # ASCIIPLOT of final action distributions in memory
        # if plot_action_density:

    return policy, total_reward


if __name__ == '__main__':
    fire.Fire(train)