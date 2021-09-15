import shutil
import glob
import os

from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from noboard.csv import SummaryWriter

import explorationlib
from explorationlib.local_gym import ScentGrid

from explorationlib.agent import DeterministicWSLSGrid
from explorationlib.agent import CriticGrid
from explorationlib.agent import SoftmaxActor
from explorationlib.agent import DiffusionGrid
from explorationlib.agent import GradientDiffusionGrid
from explorationlib.agent import AccumulatorGradientGrid
from explorationlib.agent import AccumulatorInfoGrid
from explorationlib.agent import ActorCriticGrid

from explorationlib.run import experiment
from explorationlib.util import select_exp
from explorationlib.util import load
from explorationlib.util import save

from explorationlib.local_gym import uniform_targets
from explorationlib.local_gym import constant_values
from explorationlib.local_gym import ScentGrid
from explorationlib.local_gym import create_grid_scent
from explorationlib.local_gym import add_noise
from explorationlib.local_gym import create_grid_scent_patches as create_patches

from explorationlib.plot import plot_position2d
from explorationlib.plot import plot_length_hist
from explorationlib.plot import plot_length
from explorationlib.plot import plot_targets2d
from explorationlib.plot import plot_scent_grid
from explorationlib.plot import plot_targets2d

from explorationlib.score import total_reward
from explorationlib.score import total_info_value
from explorationlib.score import search_efficiency
from explorationlib.score import num_death

from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint


class RandomScentGrid(ScentGrid):
    """ScentGrid with random targets/scemts."""
    def __init__(self):
        # Meta init
        super().__init__()

        # Init rest of Env
        self.noise_sigma = 1.0
        self.p_targets = 1.0
        self.num_targets = 20
        self.target_boundary = (10, 10)

        # Env space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(2)

        # Targets/scents/etc
        self.seed()
        self.reset()

    def reset(self):
        # Create Targets
        self.targets = uniform_targets(self.num_targets, self.target_boundary,
                                       self.np_random)
        self.values = constant_values(self.targets, 1)

        # Create 'Scent'
        scents = []
        for _ in range(len(self.targets)):
            coord, scent = create_patches(self.target_boundary,
                                          p=1.0,
                                          amplitude=1,
                                          sigma=2)
            scents.append(scent)
        self.add_scents(self.targets,
                        self.values,
                        coord,
                        scents,
                        noise_sigma=self.noise_sigma)

        # Begin in the center
        self.state = np.zeros(2)
        self.reward = 0.0
        self.last()

    def render(self, mode='human', close=False):
        pass


def run(agent_name="wsls",
         num_episodes=10,
         num_steps=200,
         lr=0.1,
         gamma=0.1,
         boredom=0.001,
         seed_value=None,
         write_to_disk=True,
         log_dir=None,
         output=True):

    # -- Init --
    # Worlds
    env = RandomScentGrid()

    # Agent 
    # (Make some hard and fixed choices)
    if agent_name == "wsls":
        agent = DeterministicWSLSGrid(lr=lr, gamma=gamma, boredom=boredom)
    elif agent_name == "diffusion":
        agent = DiffusionGrid(min_length=1, scale=1)
    elif agent_name == "sniff":
        agent = GradientDiffusionGrid(min_length=1,
                                        scale=1.0,
                                        p_neg=1,
                                        p_pos=0.0)
    elif agent_name == "entropy":
        agent = AccumulatorInfoGrid(min_length=1,
                                    max_steps=1,
                                    drift_rate=1,
                                    threshold=3,
                                    accumulate_sigma=1)
    elif agent_name == "softmax":
        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        critic = CriticGrid(default_value=0.5)
        actor = SoftmaxActor(num_actions=len(possible_actions),
                                actions=possible_actions,
                                beta=6)
        agent = ActorCriticGrid(actor, critic, lr=lr, gamma=0.1)
    else:
        raise ValueError("agent not known")

    # Re(set) seeds
    agent.seed(seed_value + n)
    agent.reset()
    env.seed(seed_value + n)
    env.reset()

    # Run
    results = experiment(f"{agent_name}",
                        agent,
                        env,
                        num_steps=num_steps,
                        num_experiments=num_episodes,
                        dump=False,
                        split_state=True,
                        seed=seed_value)

    # -- Write complete results? ---
    # Note: The exploraationlib fmt varies from the infomercial fmt used
    # elsewhere. Here I align them, as best I can.
    if write_to_disk:
        # Inir
        writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)
        
        # Extract using explib fns.
        total_Rs = total_reward(results)
        total_Es = total_info_value(results)
        total_deaths = num_death(results)
        total_effs = search_efficiency(results)
        
        # Write 'em into writer
        for n in range(num_episodes):
            writer.add_scalar("total_E", total_Es[n], n)
            writer.add_scalar("total_R", total_Rs[n], n)
            writer.add_scalar("death", total_deaths[n], n)
            writer.add_scalar("efficiency", total_effs[n], n)
            # Approx with the step values functions
            # with their mean
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
                  master_seed=seed_value)      
    if write_to_disk:
        save_checkpoint(summary,
                        filename=os.path.join(writer.log_dir, "result.pkl"))
                        
    if output:
        return summary
    else:
        return None


if __name__ == "__main__":
    import fire
    fire.Fire({run)
