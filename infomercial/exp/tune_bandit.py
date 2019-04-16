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


def tune_random(name,
                exp_name='beta_bandit',
                env_name='BanditOneHigh10-v0',
                num_episodes=1000,
                num_samples=10,
                num_processes=1,
                verbose=False,
                seed_value=None,
                **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""
    prng = np.random.RandomState(seed_value)

    # ------------------------------------------------------------------------
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(exp, exp_name)

    # Build the parallel callback
    trials = []

    def append_to_results(result):
        trials.append(result)

    # Setup default params
    params = dict(
        exp_func=exp_func,
        env_name=env_name,
        num_episodes=num_episodes,
        seed_value=seed_value,
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
        for k, (low, high) in config_kwargs.items():
            params["config"][k] = prng.uniform(low=low, high=high)

        # A worker gets the new sample
        workers.append(
            pool.apply_async(
                train, kwds=deepcopy(params), callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in workers:
        worker.get()
    pool.close()
    pool.join()

    # ------------------------------------------------------------------------
    # Save configs and total_R (full model data is dropped):
    best = get_best_trial(trials, 'total_R')

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


def tune_pbt(name,
             exp_name='beta_bandit',
             env_name='BanditOneHigh10-v0',
             top_threshold=0.25,
             num_iterations=2,
             num_episodes=2000,
             num_samples=10,
             num_processes=1,
             extend_episodes=False,
             verbose=False,
             seed_value=None,
             **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""
    prng = np.random.RandomState(seed_value)

    # ------------------------------------------------------------------------
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(exp, exp_name)

    # Build the parallel callback
    trials = []

    def append_to_results(result):
        trials.append(result)

    # Setup default params
    params = dict(
        exp_func=exp_func,
        env_name=env_name,
        num_episodes=num_episodes,
        seed_value=seed_value,
        config={})

    # ------------------------------------------------------------------------
    # Run first set!
    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for t in range(num_samples):

        # Reset param sample for safety
        params["config"] = {}

        # Make a new sample
        for k, (low, high) in config_kwargs.items():
            params["config"][k] = prng.uniform(low=low, high=high)

        # A worker gets the new sample
        workers.append(
            pool.apply_async(
                train, kwds=deepcopy(params), callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in workers:
        worker.get()
    pool.close()
    pool.join()
    pool.terminate()

    # ------------------------------------------------------------------------
    # Do PBT over num_iterations
    for _ in range(num_iterations):
        # Sort and save the top configs
        top_configs = {}
        for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
            if i < int(top_threshold * len(trials)):
                top_configs[i] = trial["config"]

        # Replicate and perturb
        rep_configs = {}
        for i, config in top_configs.items():
            k = i + len(top_configs)
            rep_configs[k] = {}
            for key, value in config.items():
                delta = (value * top_threshold)
                rep_configs[k][key] = prng.uniform(value - delta,
                                                   value + delta)

        print(f"top: {top_configs}")
        print(f"new: {rep_configs}")

        # Join old and new
        configs = deepcopy(top_configs)
        configs.update(rep_configs)

        # Reset trials for the next round
        trials = []

        # Extend run time
        if extend_episodes:
            params["num_episodes"] += int(
                params["num_episodes"] * top_threshold)
            print(f"Update {params['num_episodes']}")

        # Re-init the pool
        workers = []
        pool = Pool(processes=num_processes)

        # Do another round of opt, with the new config
        for t in range(len(configs)):
            # Reset param sample for safety
            params["config"] = configs[t]

            # A worker gets the new sample
            workers.append(
                pool.apply_async(
                    train, kwds=deepcopy(params), callback=append_to_results))

        # Get the worker's result (blocks until complete)
        for worker in workers:
            worker.get()
        pool.close()
        pool.join()
        pool.terminate()

    # ------------------------------------------------------------------------
    # Save configs and total_R (full model data is dropped):
    best = get_best_trial(trials, 'total_R')

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
    # Generate CL interface.
    fire.Fire({"tune_random": tune_random, "tune_pbt": tune_pbt})
