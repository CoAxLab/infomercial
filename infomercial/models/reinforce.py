"""
Asst. policy gradients, and helper functions. (In Jax).

Some code borrowed from:
https://github.com/google/jax/blob/master/examples/mnist_classifier_fromscratch.py
"""
import gym

import numpy as npnp
import numpy.random as npr

import jax
import jax.numpy as np
from jax import vmap
from jax.api import jit, grad
from jax.config import config

from functools import partial

from infomercial.models import shared
from infomercial.models.shared import step
from infomercial.models.shared import init_dense_params

# Used in numerically safe std() estimation
EPS = npnp.finfo(npnp.float32).eps.item()


def normalize_rewards(rewards, gamma=1.0):
    """Reweight and Z-score rewards (a 1d list-like)."""
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    return (rewards - np.mean(rewards) / np.std(rewards + EPS))


def loss(params, policy, transitions, gamma=1.0):
    states = np.asarray([transition[1] for transition in transitions])
    probs = vmap(partial(policy, params), states)

    rewards = np.asarray([transition[4] for transition in transitions])
    rewards = normalize_rewards(rewards, gamma=gamma)

    L = []
    for log_prob, reward in zip(np.log(probs), rewards):
        L.append(-log_prob * reward)
    return -np.sum(L)


def train(env_name,
          num_episodes=10,
          layer_sizes=None,
          param_scale=0.1,
          policy='linear_policy',
          batch_size=48,
          lr=.001,
          gamma=1.0,
          action_space='discrete',
          debug=False,
          max_steps=1000,
          seed=0):
    """Train a REINFORCEing agent."""

    # ------------------------------------------------------------------------
    # Init gym
    env = gym.make(env_name)
    env.seed(seed)

    # Init agent
    policy = getattr(shared, policy)
    params = init_dense_params(param_scale, layer_sizes)

    # ------------------------------------------------------------------------
    @jit
    def update(params, transitions):
        """SGD."""
        grads = grad(loss)(params, policy, transitions, gamma)
        return [(w - lr * dw, b - lr * db)
                for (w, b), (dw, db) in zip(params, grads)]

    # ------------------------------------------------------------------------
    # Rollouts: # (s, a, p(a), s', r)
    transitions = []
    total_reward = 0.0
    for episode in range(num_episodes):
        state = np.asarray(env.reset())
        done = False
        num_steps = 0
        while not done or (num_steps < max_steps):
            # Move
            env, action, probs, next_state, reward, done = step(
                params,
                env,
                state,
                policy=policy,
                action_space=action_space,
            )
            next_state = np.asarray(next_state)
            probs = np.asarray(probs)

            # Remember
            transitions.append((state, action, probs, next_state, reward))
            num_steps += 1

        total_reward += npnp.mean(
            [transition[4] for transition in transitions])
        if debug:
            print(f">>> Iteration {episode}.")
            print(f">>> Total reward {total_reward}.")

        # --------------------------------------------------------------------
        # Learn?
        if len(transitions) > batch_size:
            if debug:  # Needs to print before update...
                print(f">>> Loss {loss(params, transitions, gamma=gamma)}")

            params = update(params, transitions)

    return params, total_reward