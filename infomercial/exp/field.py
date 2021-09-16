import os
import numpy as np

from copy import deepcopy

from noboard.csv import SummaryWriter
from infomercial.utils import save_checkpoint

from explorationlib.local_gym import ScentGrid
from explorationlib.agent import DeterministicWSLSGrid
from explorationlib.agent import CriticGrid
from explorationlib.agent import SoftmaxActor
from explorationlib.agent import DiffusionGrid
from explorationlib.agent import GradientDiffusionGrid
from explorationlib.agent import AccumulatorInfoGrid
from explorationlib.agent import ActorCriticGrid
from explorationlib.run import experiment
from explorationlib.local_gym import uniform_targets
from explorationlib.local_gym import constant_values
from explorationlib.local_gym import ScentGrid
from explorationlib.local_gym import create_grid_scent_patches
from explorationlib.score import total_reward
from explorationlib.score import total_info_value
from explorationlib.score import search_efficiency
from explorationlib.score import num_death


def wsls(num_episodes=10,
         num_steps=200,
         lr=0.1,
         gamma=0.1,
         boredom=0.001,
         master_seed=None,
         write_to_disk=True,
         log_dir=None,
         output=True):

    agent_name = "wsls"
    agent = DeterministicWSLSGrid(lr=lr, gamma=gamma, boredom=boredom)

    return _run(agent, agent_name, num_episodes, num_steps, master_seed,
                write_to_disk, log_dir, output)


def diffusion(num_episodes=10,
              num_steps=200,
              min_length=1,
              scale=1,
              master_seed=None,
              write_to_disk=True,
              log_dir=None,
              output=True):

    agent_name = "diffusion"
    agent = DiffusionGrid(min_length=min_length, scale=scale)

    return _run(agent, agent_name, num_episodes, num_steps, master_seed,
                write_to_disk, log_dir, output)


def chemotaxis(num_episodes=10,
               num_steps=200,
               min_length=1,
               scale=1,
               p_neg=1,
               p_pos=0.0,
               master_seed=None,
               write_to_disk=True,
               log_dir=None,
               output=True):

    agent_name = "chemotaxis"
    agent = GradientDiffusionGrid(min_length=min_length,
                                  scale=scale,
                                  p_neg=p_neg,
                                  p_pos=p_pos)

    return _run(agent, agent_name, num_episodes, num_steps, master_seed,
                write_to_disk, log_dir, output)


def entropy(num_episodes=10,
            num_steps=200,
            min_length=1,
            max_steps=1,
            drift_rate=1,
            threshold=3,
            accumulate_sigma=1,
            master_seed=None,
            write_to_disk=True,
            log_dir=None,
            output=True):

    agent_name = "entropy"
    agent = AccumulatorInfoGrid(min_length=min_length,
                                max_steps=max_steps,
                                drift_rate=drift_rate,
                                threshold=threshold,
                                accumulate_sigma=accumulate_sigma)

    return _run(agent, agent_name, num_episodes, num_steps, master_seed,
                write_to_disk, log_dir, output)


def softmax(num_episodes=10,
            num_steps=200,
            lr=0.1,
            temp=6,
            gamma=0.1,
            master_seed=None,
            write_to_disk=True,
            log_dir=None,
            output=True):

    agent_name = "softmax"
    possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    critic = CriticGrid(default_value=0.5)
    actor = SoftmaxActor(num_actions=len(possible_actions),
                         actions=possible_actions,
                         beta=temp)
    agent = ActorCriticGrid(actor, critic, lr=lr, gamma=gamma)

    return _run(agent, agent_name, num_episodes, num_steps, master_seed,
                write_to_disk, log_dir, output)


def _run(agent, agent_name, num_episodes, num_steps, master_seed,
         write_to_disk, log_dir, output):

    # -- Init --
    num_targets = 20
    target_boundary = (10, 10)  # Field size
    reward = 1  # Target value
    p_scent = 1.0  # Prob detection
    noise_sigma = 1.0  # Sensor noise

    # Targets
    prng = np.random.RandomState(master_seed)
    targets = uniform_targets(num_targets, target_boundary, prng=prng)
    values = constant_values(targets, reward)

    # Scents
    scents = []
    for _ in range(len(targets)):
        coord, scent = create_grid_scent_patches(
            target_boundary,
            p=p_scent,
            amplitude=1,  # Scent width
            sigma=2)  # Scent mag.
        scents.append(scent)

    # Env
    env = ScentGrid(mode=None)
    env.seed(master_seed)
    env.add_scents(targets, values, coord, scents, noise_sigma=noise_sigma)

    # -- Run --
    results = experiment(f"{agent_name}",
                         agent,
                         env,
                         num_steps=num_steps,
                         num_experiments=num_episodes,
                         dump=False,
                         split_state=True,
                         seed=master_seed)

    # -- Write complete results? ---
    # Note: The exploraationlib fmt varies from the infomercial fmt used
    # elsewhere. Here I align them, as best I can.

    # Get totals
    total_Rs = total_reward(results)
    total_Es = total_info_value(results)
    total_deaths = num_death(results)
    total_effs = search_efficiency(results)

    # Write?
    if write_to_disk:
        # Inir
        writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

        # Write 'em
        for n in range(num_episodes):
            writer.add_scalar("total_E", total_Es[n], n)
            writer.add_scalar("total_R", total_Rs[n], n)
            writer.add_scalar("num_death", total_deaths, n)
            writer.add_scalar("efficiency", total_effs[n], n)
            # Approx total step value functions
            log = results[n]
            writer.add_scalar("value_R", np.mean(log["agent_reward_value"]), n)
            writer.add_scalar("value_E", np.mean(log["agent_info_value"]), n)

        # Clean
        writer.close()

    # -- Summarize --
    summary = dict(env_name="RandomScentGrid",
                   agent_name=agent_name,
                   agent=deepcopy(agent),
                   env=deepcopy(env),
                   num_episodes=num_episodes,
                   total_E=total_Es[-1],
                   total_R=total_Rs[-1],
                   master_seed=master_seed)
    if write_to_disk:
        save_checkpoint(summary,
                        filename=os.path.join(writer.log_dir, "result.pkl"))

    if output:
        return summary
    else:
        return None


if __name__ == "__main__":
    import fire
    fire.Fire({
        "wsls": wsls,
        "softmax": softmax,
        "diffusion": diffusion,
        "chemotaxis": chemotaxis,
        "entropy": entropy
    })
