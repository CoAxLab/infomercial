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


def run(agent="wsls",
         num_episodes=10,
         num_steps=200,
         num_experiments=10,
         lr=0.1,
         gamma=0.1,
         boredom=0.001,
         seed_value=None,
         write_to_disk=True,
         log_dir=None,
         output=True):

    # -- Init --
    env = RandomScentGrid()

    # -- Run ! --
    results = []

    # TODO - HERE - should I only be run num_experiments as num_episodes
    # because anything else is not consistent w/ the data in bandits.
    
    for n in range(num_episodes):
        # - Imprelentation note: -
        # Agents defined every iteration just
        # in caase agent.reset() is not a complete.
        # reset. It might not be in exploirationlib
        # agnets.
        if agent == "wsls":
            model = DeterministicWSLSGrid(lr=lr, gamma=gamma, boredom=boredom)
        elif agent == "diffusion":
            model = DiffusionGrid(min_length=min_length, scale=1)
        elif agent == "sniff":
            min_length = 1
            model = GradientDiffusionGrid(min_length=1,
                                          scale=1.0,
                                          p_neg=1,
                                          p_pos=0.0)
        elif agent == "entropy":
            model = AccumulatorInfoGrid(min_length=1,
                                        max_steps=1,
                                        drift_rate=1,
                                        threshold=3,
                                        accumulate_sigma=1)
        elif agent == "softmax":
            possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            critic = CriticGrid(default_value=0.5)
            actor = SoftmaxActor(num_actions=4,
                                 actions=possible_actions,
                                 beta=6)
            model = ActorCriticGrid(actor, critic, lr=0.1, gamma=0.1)
        else:
            raise ValueError("agent not known")

        # Set seeds
        model.seed(seed_value + n)
        env.seed(seed_value + n)

        # Reset
        model.reset()
        env.reset()

        # Run
        result = experiment(f"{agent}",
                            model,
                            env,
                            num_steps=num_steps,
                            num_experiments=num_experiments,
                            dump=False,
                            split_state=True,
                            seed=seed_value)
        results.append(deepcopy(result))

    # -- Build the final result, and save or return it ---
    # Extract data to write
    if write_to_disk:
        result = None # safety
        writer = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)
        for n, result in enumerate(results):
            total_R = total_reward(result)
            writer.add_scalar("total_R", total_R, n)

            # Get state/value
        # Log data
        writer.add_scalar("state", int(state), n)
        writer.add_scalar("action", action, n)
        writer.add_scalar("regret", regret, n)
        writer.add_scalar("bonus", novelty_bonus, n)
        writer.add_scalar("score_E", E_t, n)
        writer.add_scalar("score_R", R_t, n)
        writer.add_scalar("value_ER", critic(action), n)
        writer.add_scalar("value_R", critic(action), n)

        total_E += E_t
        total_R += R_t
        total_regret += regret
        writer.add_scalar("total_regret", total_regret, n)
        writer.add_scalar("total_E", total_E, n)
        writer.add_scalar("total_R", total_R, n)
        writer.add_scalar("p_bests", num_best / (n + 1), n)
    
    writer.close()

    # -- Summarize ressults --
    summary = dict(env_name="RandomScentGrid",
                  agent_name=agent,
                  model=deepcopy(model),
                  env=deepcopy(env),
                  num_episodes=num_episodes,
                  total_E=total_E,
                  total_R=total_R,
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
