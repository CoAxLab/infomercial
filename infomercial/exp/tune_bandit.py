import fire
import ray
import os
from ray.tune import sample_from
from ray.tune import grid_search
from ray.tune import function
from ray.tune import run as ray_run
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining

import numpy as np
from infomercial import exp
# from infomercial.exp import beta_bandit
# from infomercial.exp import epsilon_bandit
from infomercial.exp.meta_bandit import save_checkpoint
from infomercial.exp.meta_bandit import load_checkpoint

from functools import partial


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))


def get_sorted_trials(trial_list, metric):
    return sorted(
        trial_list,
        key=lambda trial: trial.last_result.get(metric, 0),
        reverse=True)


def get_best_result(trial_list, metric):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric).last_result[metric]}


class TuneBanditBase(Trainable):
    def _setup(self, config):
        self.config = config
        self.iteration = 0
        self.result = None

    def _save(self, filename):
        return save_checkpoint(self.result, filename=filename)

    def _restore(self, filename):
        return load_checkpoint(filename=filename)

    def reset_config(self, new_config):
        self.config = new_config
        return True


def run(name,
        exp_name='beta_bandit',
        env_name='BanditOneHigh10-v0',
        num_episodes=1000,
        num_samples=10,
        verbose=False,
        **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""

    # ------------------------------------------------------------------------
    # Init

    # Separate name from path
    path, name = os.path.split(name)

    # Build the config
    config = {}
    keys = sorted(list(config_kwargs.keys()))
    for k in keys:
        begin, end = config_kwargs[k]
        if verbose:
            print(f">>> Sampling {k} from {begin}-{end}")
        config[k] = sample_from(lambda spec: np.random.uniform(begin, end))

    # Define the final Trainable.
    class Tuner(TuneBanditBase):
        def _train(self):
            exp_func = getattr(exp, exp_name)
            self.result = exp_func(
                env_name=env_name,
                num_episodes=num_episodes,
                seed_value=None,
                **self.config)
            self.result.update({"iteration": self.iteration})
            self.iteration += 1

            return self.result

    # ------------------------------------------------------------------------
    # Opt!
    trials = ray_run(
        Tuner,
        name=name,
        local_dir=path,
        num_samples=num_samples,
        config=config,
        stop={"iteration": 1},
        verbose=False)
    best = get_best_trial(trials, 'total_R')

    # ------------------------------------------------------------------------
    # Re-save the interesting parts:
    # Best trial
    best_config = best.config
    best_config.update(get_best_result(trials, 'total_R'))
    save_checkpoint(
        best_config, filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save a sum of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
        sorted_configs[i] = trial.config
        sorted_configs[i].update({"total_R": trial.last_result["total_R"]})
    save_checkpoint(
        sorted_configs, filename=os.path.join(path, name + "_sorted.pkl"))

    return best, trials


if __name__ == "__main__":
    # Get ray goin before the CL runs
    ray.init()

    # Generate CL interface.
    fire.Fire(run)
