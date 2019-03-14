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

from infomercial import memory as mem
from infomercial.utils import sample_action
from infomercial.utils import best_action
from infomercial.utils import normal_action
from infomercial.utils import save_checkpoint
from infomercial.utils import ZFilter
from infomercial.utils import Oracle
from infomercial.rollouts import ModulusMemory

from infomercial.models import Actor3Sigma
from infomercial.models import Critic3
from infomercial.models import Lookup
from infomercial.utils import build_hyperparameters
from infomercial.utils import create_envs

# PPO
from infomercial.agents.ppo.core import train_model as train_model_ppo
from infomercial.agents.ppo.core import test_model as test_model_ppo
from infomercial.agents.ppo.core import update_current_observation

# INFO
from infomercial.agents.info.core import train_model as train_model_info
from infomercial.agents.info.core import test_model as test_model_info


def run_meta(env_name='MountainCarContinuous-v0',
             update_every=100,
             save=None,
             progress=True,
             cuda=False,
             debug=False,
             render=False,
             **algorithm_hyperparameters):

    pass


def run_info(
        env_name='MountainCarContinuous-v0',
        update_every=100,
        save=None,
        progress=True,
        cuda=False,
        use_torch=False,
        debug=False,
        render=False,
        cond_on_state=False,  # will need this for bandits, others?
        batch_mode=False,
        memory_name='Count',
        num_episodes=100,
        num_memories=200,
        num_processes=1,
        default_value=0,
        **memory_kwargs):

    # ------------------------------------------------------------------------
    # Setup the world
    device = torch.device('cuda') if cuda else torch.device('cpu')
    prng = np.random.RandomState(hp.seed_value)

    # Wrap the envs in a oracle to count the total number of gym.step()
    # calls across all envs/processes
    envs = create_envs(env_name, hp.num_processes, hp)
    envs = Oracle(envs, step_value=hp.num_processes)

    test_env = gym.make(env_name)
    test_env.seed(hp.seed_value)

    # State and action shapes
    num_inputs = envs.observation_space.shape[0]
    num_actions = envs.action_space.shape[0]

    # ------------------------------------------------------------------------
    hp = build_hyperparameters(
        env_name=env_name,
        num_inputs=num_inputs,
        num_actions=num_actions,
        update_every=update_every,
        save=save,
        progress=progress,
        cuda=cuda,
        device=device,
        use_torch=use_torch,
        debug=debug,
        render=render,
        cond_on_state=cond_on_state,  # will need this for bandits, others?
        batch_mode=batch_mode,
        memory_name=memory_name,
        num_episodes=num_episodes,
        num_memories=num_memories,
        num_processes=num_processes,
        default_value=default_value)

    # ------------------------------------------------------------------------
    # Agents and memories
    actor = np.argmax
    critic = Lookup(hp.num_actions, hp.default_value)
    rollout = ModulusMemory(hp.num_memories)
    memory = getattr(mem, hp.memory_name)(**memory_kwargs)

    # ------------------------------------------------------------------------
    # Play many games
    episode = 0
    episodes_scores = []
    total_steps = []
    for episode in range(hp.num_episodes):
        # Re-init
        state = envs.reset()
        rollout.reset()
        score = 0
        scores = []

        # Run until the memory is full
        for n in range(hp.num_memories):
            # Choose an action
            Q_n = critic(state)
            action = actor(Q_n)

            # Do the action
            next_state, reward, done, _ = envs.step(action)

            # Process outcome
            reward = np.expand_dims(np.stack(reward), 1)
            mask = np.zeros(*done.shape)
            for i, d in enumerate(done):
                if d:
                    mask[i] = 1.0

            # Save the rollout
            rollout.push(state, action, reward, mask)

            # Save the score / step
            score += np.mean(reward.numpy())
            scores.append(score)

            # Shift state
            state = next_state

            # Sequential learning mode?
            if not batch_mode:
                train_model_info()

        # --------------------------------------------------------------------
        # Batch learning mode?
        if batch_mode:
            train_model_info()

        # Test the learned....
        # We do this in a fresh env to get consistent scores, which can be
        # weird w/ gymvec's async.
        _, test_scores = test_model_info(
            actor, critic, test_env, hp, render=hp.render)
        score_avg = np.mean(test_scores)

        # Save results
        episodes_scores.append(score_avg)
        total_steps.append(envs.total_steps)

        # --------------------------------------------------------------------
        if progress:
            print(">>> Episode {} avg. score {}".format(n, score_avg))

        if (save is not None) and (n % update_every) == 0:
            f_name = save + "_ep_{}.pytorch.tar".format(n)
            save_checkpoint({
                'hyperparameters': hp.state_dict(),
                'critic': critic.state_dict(),
                'score': score_avg
            },
                            filename=f_name)

    return total_steps, episodes_scores


def run_ppo(
        env_name='MountainCarContinuous-v0',
        update_every=100,
        save=None,
        progress=True,
        cuda=False,
        debug=False,
        render=False,
        gamma=0.99,
        lam=0.98,
        actor_hidden1=64,
        actor_hidden2=64,
        actor_hidden3=64,
        critic_hidden1=64,
        critic_lr=0.0003,
        actor_lr=0.0003,
        batch_size=64,
        l2_rate=0.001,
        clip_param=0.2,
        num_episodes=100,
        num_memories=200,
        num_processes=1,
        num_stack=1,
        num_training_epochs=10,
        clip_actions=True,
        clip_std=1.0,  #0.25
        seed_value=3959):

    # ------------------------------------------------------------------------
    hp = build_hyperparameters(
        gamma=gamma,
        lam=lam,
        actor_hidden1=actor_hidden1,
        actor_hidden2=actor_hidden2,
        actor_hidden3=actor_hidden3,
        critic_hidden1=critic_hidden1,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        batch_size=batch_size,
        l2_rate=l2_rate,
        clip_param=clip_param,
        num_training_epochs=num_training_epochs,
        num_episodes=num_episodes,
        num_memories=num_memories,
        num_processes=num_processes,
        num_stack=num_stack,
        clip_actions=clip_actions,
        clip_std=clip_std,
        seed_value=seed_value)

    device = torch.device('cuda') if cuda else torch.device('cpu')

    # ------------------------------------------------------------------------
    # Setup the world
    envs = create_envs(env_name, hp.num_processes, hp)

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

        # Reset the worlds
        state = envs.reset()

        # -
        for n_m in range(hp.num_memories):
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
        train_model_ppo(actor, critic, memory, actor_optim, critic_optim,
                        device, hp)

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
