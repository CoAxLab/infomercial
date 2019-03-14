def kl_divergence(new, old):
    # thin wrap of scipy
    pass


def train_model(actor,
                critic,
                memory,
                rollout,
                actor_optim,
                critic_optim,
                memory_optim,
                device,
                hp,
                state_space=None):

    # parse instance of memory: Memory (native python) or Pytorch
    # vec the memory? Set up state parsing accordingly

    for m in rollout.memories():

        # Sample the untrained memory with state_space -> p_old
        # Update the memory (?)
        # Sample the trained memory with state_space -< p_new
        # Estimate E_m
        E = kl_divergence(p_new, p_old)

        # Update the critic with E_M
        # Update the actor?

        pass


def test_model(policy, env, hp, render=False):
    raise NotImplementedError("TODO")

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
