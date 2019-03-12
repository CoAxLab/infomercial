"""Test games with flowing actions."""
import os
import errno

from collections import deque

import gym_vecenv
import gym
from gym import wrappers

import numpy as np
import torch
import torch.optim as optim

from infomercial.utils import sample_action
from infomercial.utils import best_action
from infomercial.utils import normal_action
from infomercial.utils import save_checkpoint
from infomercial.utils import ZFilter
from infomercial.utils import Oracle
from infomercial.rollouts import ModulusMemory

from infomercial.models import Actor3Sigma
from infomercial.models import Critic3

# PPO
from infomercial.agents.ppo.core import train_model as train_model_ppo
from infomercial.agents.ppo.core import test_model as test_model_ppo
from infomercial.agents.ppo.core import Hyperparameters as Hyperparameters_PPO
from infomercial.agents.ppo.core import create_envs as create_envs_ppo
from infomercial.agents.ppo.core import update_current_observation


def run_ppo(env_name='MountainCarContinuous-v0',
            update_every=100,
            save=None,
            progress=True,
            cuda=False,
            debug=False,
            render=False,
            **algorithm_hyperparameters):

    # ------------------------------------------------------------------------
    device = torch.device('cuda') if cuda else torch.device('cpu')

    # and its hyperparams
    hp = Hyperparameters_PPO()
    for k, v in algorithm_hyperparameters.items():
        setattr(hp, k, v)

    # ------------------------------------------------------------------------
    # Setup the world
    envs = create_envs_ppo(env_name, hp.num_processes, hp)

    # Wrap the envs in a oracle to count the total number of gym.step()
    # calls across all envs/processes
    envs = Oracle(envs, step_value=hp.num_processes)

    prng = np.random.RandomState(hp.seed_value)

    test_env = gym.make(env_name)
    test_env.seed(hp.seed_value)

    # State and action shapes
    num_inputs = envs.observation_space.shape[0]
    num_actions = envs.action_space.shape[0]
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * hp.num_stack, *obs_shape[1:])

    # ------------------------------------------------------------------------
    # Init the rollout memory
    memory = ModulusMemory(hp.num_memories)

    # ------------------------------------------------------------------------
    # Actor-critic init
    actor = Actor3Sigma(num_inputs, num_actions, hp, max_std=hp.clip_std)
    critic = Critic3(num_inputs, hp)

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(
        critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    critic.to(device)
    actor.to(device)

    # ------------------------------------------------------------------------
    # Play many games
    episode = 0
    episodes_scores = []
    total_steps = []
    current_observation = torch.zeros(hp.num_processes, *obs_shape)
    for n_e in range(hp.num_episodes):
        # Re-init
        actor.eval()
        critic.eval()
        memory.reset()
        score = 0
        scores = []
        steps = 0
        episode += 1

        # -
        for n_m in range(hp.num_memories):
            # Reset the worlds
            state = envs.reset()
            current_observation = update_current_observation(
                hp, envs, state, current_observation)

            # Choose an action
            mu, std, _ = actor(torch.tensor(state).float())
            action = normal_action(mu, std)
            action_std = std.clone().detach().numpy().flatten()
            if hp.clip_actions:
                action = np.clip(action, envs.action_space.low,
                                 envs.action_space.high)

            # Do the action
            next_state, reward, done, info = envs.step(action)

            # Process outcome
            reward = np.expand_dims(np.stack(reward), 1)
            reward = torch.from_numpy(reward).float()
            mask = np.zeros(*done.shape)
            for i, d in enumerate(done):
                if d:
                    mask[i] = 1.0

            # Save/update
            memory.push(state, action, reward, mask, action_std)
            score += np.mean(reward.numpy())
            scores.append(score)

            # Shift
            state = next_state

            # -
            if debug and (n_m % update_every) == 0:
                print(">>> Mem. {}".format(n_m))
                print(">>> Last score {}".format(score))
                print(">>> Mu, Sigma ({}, {})".format(mu.tolist(),
                                                      std.tolist()))

        # Total number of calls to the envs (as judged by the oracle)
        total_steps.append(envs.total_steps)

        # --------------------------------------------------------------------
        # Learn!
        actor.train()
        critic.train()
        train_model_ppo(
            actor,
            critic,
            memory,
            actor_optim,
            critic_optim,
            device,
            hp,
            num_training_epochs=hp.num_training_epochs)

        # Test the learned. Do this in a fresh env to get consistent
        # scores, which can be weird w/ gymvec's async.
        _, test_scores = test_model_ppo(actor, test_env, hp, render=render)
        score_avg = np.mean(test_scores)
        episodes_scores.append(score_avg)

        if progress:
            print(">>> Episode {} avg. score {}".format(n_e, score_avg))

        # --------------------------------------------------------------------
        if (save is not None) and (n_e % update_every) == 0:
            f_name = save + "_ep_{}.pytorch.tar".format(n_e)
            save_checkpoint({
                'policy': policy.state_dict(),
                'score': score_avg
            },
                            filename=f_name)

    return total_steps, episodes_scores
