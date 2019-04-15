import fire
import ray
import os
from copy import deepcopy
# from ray.tune import sample_from
# from ray.tune import function
# from ray.tune import grid_search
# from ray.tune import function
# from ray.tune import run as ray_run
# from ray.tune import Trainable
# from ray.tune.schedulers import PopulationBasedTraining
from multiprocessing import Pool

import numpy as np
from infomercial import exp
# from infomercial.exp import beta_bandit
# from infomercial.exp import epsilon_bandit
from infomercial.exp.meta_bandit import save_checkpoint
from infomercial.exp.meta_bandit import load_checkpoint

from functools import partial


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial[metric])


def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial[metric], reverse=True)


def get_best_result(trial_list, metric):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric)[metric]}


def train(exp_func=None,
          env_name=None,
          num_episodes=None,
          seed_value=None,
          config=None):

    # Run
    trial = exp_func(
        env_name=env_name,
        num_episodes=num_episodes,
        seed_value=seed_value,
        **config)

    # Save metadata
    trial.update({
        "config": config,
        "env_name": env_name,
        "num_episodes": num_episodes,
        "seed_value": seed_value
    })

    return trial


def run(name,
        exp_name='beta_bandit',
        env_name='BanditOneHigh10-v0',
        num_episodes=1000,
        num_samples=10,
        num_processes=1,
        verbose=False,
        **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""

    # ------------------------------------------------------------------------
    # Init

    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(exp, exp_name)

    # Build the sampling config
    config = {}
    keys = sorted(list(config_kwargs.keys()))
    for k in keys:
        begin, end = config_kwargs[k]
        config[k] = partial(np.random.uniform, low=begin, high=end)

        if verbose:
            print(f">>> Sampling {k} from {begin}-{end}")

    # Build the parallel callback
    trials = []

    def append_to_results(result):
        trials.append(result)

    # Setup default params
    params = dict(
        exp_func=exp_func,
        env_name=env_name,
        num_episodes=num_episodes,
        seed_value=None,
        config={})

    # ------------------------------------------------------------------------
    # Run!

    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for _ in range(num_samples):
        # Reset param sample for safety
        params["config"] = {}

        # Make a new sample
        for k, f in config.items():
            params["config"][k] = f()

        # A worker gets the new sample
        workers.append(
            pool.apply_async(train, kwds=params, callback=append_to_results))

    # Get the worker's result (blocks until all complete)
    for worker in workers:
        worker.get()

    pool.close()
    pool.join()

    best = get_best_trial(trials, 'total_R')

    # ------------------------------------------------------------------------
    # Save interesting configs (full model data is dropped):

    # Best trial config
    best_config = best["config"]
    best_config.update(get_best_result(trials, 'total_R'))
    save_checkpoint(
        best_config, filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({"total_R": trial["total_R"]})
    save_checkpoint(
        sorted_configs, filename=os.path.join(path, name + "_sorted.pkl"))

    return best, trials


if __name__ == "__main__":
    # Get ray goin before the CL runs
    # ray.init()

    # Generate CL interface.
    fire.Fire(run)
