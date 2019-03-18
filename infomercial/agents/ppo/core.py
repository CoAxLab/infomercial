import torch
import gym_vecenv
import numpy as np

from slapdash.utils import log_probability
from slapdash.envs import make_env
from slapdash.envs import make_env_vec
from slapdash.utils import get_action


def get_returns(rewards, masks, values, hp):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = (
            rewards[t] + hp.gamma * previous_value * masks[t] - values.data[t])
        running_advants = (
            running_tderror + hp.gamma * hp.lam * running_advants * masks[t])

        returns[t] = running_returns
        previous_value = values.data[t]
        advantages[t] = running_advants

    advantages = (advantages - advantages.mean()) / advantages.std()
    return returns, advantages


def surrogate_loss(actor, advantages, states, old_policy, actions, index):
    mu, std, logstd = actor(states)
    new_policy = log_probability(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advantages
    return surrogate, ratio


def vectorize(rollout, device):
    states, actions, rewards, masks, action_stds = zip(*rollout.memory)
    states = torch.from_numpy(np.vstack(states)).float().to(device)
    actions = torch.from_numpy(np.vstack(actions)).float().to(device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
    masks = torch.from_numpy(np.concatenate(masks)).float().to(device)
    action_stds = torch.from_numpy(np.vstack(action_stds)).float().to(device)

    return states, actions, rewards, masks, action_stds


def update_current_observation(hp, envs, state, current_observation):
    shape_dim0 = envs.observation_space.shape[0]
    state = torch.from_numpy(state).float()
    if hp.num_stack > 1:
        current_observation[:, :-shape_dim0] = current_observation[:,
                                                                   shape_dim0:]
    current_observation[:, -shape_dim0:] = state

    return current_observation


def train_model(actor, critic, rollout, actor_optim, critic_optim, device, hp):
    """Train an a2c PPO model."""
    # ------------------------------------------------------------------------
    states, actions, rewards, masks, _ = vectorize(rollout, device)

    # ------------------------------------------------------------------------
    # step 1: get returns and GAEs and log probability of old policy
    values = critic(states)

    returns, advantages = get_returns(rewards, masks, values, hp)
    mu, std, logstd = actor(states)
    old_policy = log_probability(actions, mu, std, logstd)
    old_values = critic(states)

    criterion = torch.nn.MSELoss()
    n = states.shape[0]
    arr = np.arange(n)

    # ------------------------------------------------------------------------
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(hp.num_training_epochs):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i:hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = states[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advantages_samples = advantages.unsqueeze(1)[batch_index]
            actions_samples = actions[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advantages_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -hp.clip_param,
                                         hp.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio, 1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advantages_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


def test_model(policy, env, hp, render=False):
    # Make a fresh world; init
    state = env.reset()
    score = 0
    step = 0
    scores = []
    episodes = []
    done = False

    # Play until done
    while not done:
        if render:
            env.render()

        mu, std, _ = policy(torch.tensor(state).float())
        action = get_action(mu, std)
        action_std = std.clone().detach().numpy().flatten()
        if hp.clip_actions:
            action = np.clip(action, env.action_space.low,
                             env.action_space.high)

        # Do the action
        next_state, reward, done, info = env.step(action)

        # Update score
        score += reward
        scores.append(score)

        # Count steps
        episodes.append(step)
        step += 1

        # Shift
        state = next_state

    return episodes, scores
