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
from infomercial.util import save_checkpoint

EPS = np.finfo(np.float32).eps.item()


def select_action(policy, state, mode='Categorical'):
    # Get the current policy pi_s
    state = state.float().unsqueeze(0)
    pi_s = policy(state)

    # Use pi_s to make an action, using any dist in torch.
    # The dist should match the policy, of course.
    Dist = getattr(torch.distributions, mode)
    m = Dist(*pi_s)
    action = m.sample()

    # The policy agent keeps track of it's own training data.
    policy.log_probs.append(m.log_prob(action))

    return policy, action.item()


def update(policy, optimizer, gamma=1.0):
    loss = []

    # Re-weight rewards with gamma. (Mem. eff. if ugly.)
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)

    # Norm rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

    # Log prob loss!
    for log_prob, reward in zip(policy.log_probs, rewards):
        loss.append(-log_prob * reward)

    # Backprop teach us everything.
    optimizer.zero_grad()
    loss = torch.tensor(loss).sum()
    loss.backward()
    optimizer.step()

    # Wipe the agent's memory
    del policy.rewards[:]
    del policy.log_probs[:]

    return policy, optimizer, loss


def train(env_name='BanditTwoArmedDeterministicFixed',
          num_episodes=100,
          batch_size=48,
          lr=0.001,
          learn=True,
          save=None,
          progress=True,
          debug=False,
          log_interval=1,
          render=False,
          seed=349,
          gamma=1.0,
          action_mode='Categorical',
          model_name='LinearCategorical',
          **model_hyperparameters):

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

    # ------------------------------------------------------------------------
    # Run some games!
    total_reward = 0.0
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state)

        if debug:
            print(f"--- Episode {episode} ---")
            print(f">>> Initial state {state}")

        done = False
        while not done:  # Don't infinite loop while learning
            policy, action = select_action(policy, state, mode=action_mode)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            state = torch.tensor(state)

            if debug:
                print(f">>> Action {action}")
                print(f">>> p(action) {np.exp(policy.log_probs[-1].detach())}")
                print(f">>> Next state {state}")
                print(f">>> Reward {reward}")

            if render:
                env.render()

        avg_r = np.mean(policy.rewards)
        total_reward += avg_r

        loss = None
        if learn and (len(policy.rewards) >= batch_size):
            policy, optimizer, loss = update(policy, optimizer, gamma=gamma)

        if ((episode % log_interval) == 0 and progress) or debug:
            print(f">>> Loss {loss}")
            print(f">>> Last reward {avg_r}")
            print(f">>> Total reward {total_reward}")

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

    return policy, total_reward


if __name__ == '__main__':
    fire.Fire(train)