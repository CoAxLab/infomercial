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


def get_configs(trial_list):
    """Extract configs"""

    return [trial["config"] for trial in trial_list]


def get_metrics(trial_list, metric):
    """Extract metric"""
    return [trial[metric] for trial in trial_list]


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
    top_threshold = 0.5  # Fix to hold pop size const, as per PBT.

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


def tune_replicator(name,
                    exp_name='beta_bandit',
                    env_name='BanditOneHigh10-v0',
                    num_iterations=2,
                    num_episodes=2000,
                    num_replicators=10,
                    num_processes=1,
                    perturbation=0.1,
                    extend_episodes=False,
                    verbose=False,
                    seed_value=None,
                    debug=False,
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
    # Run first population!
    population = np.ones(num_replicators) / num_replicators
    if debug: print(f">>> Intial population: {population}")

    #
    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for t in range(num_replicators):

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

    if debug: print(f">>> Example intial config{params['config']}")

    # ------------------------------------------------------------------------
    # Optimize for num_iterations
    for n in range(num_iterations):
        # --------------------------------------------------------------------
        # Replicate!
        if debug: print(f">>> Begining to replicate")
        if debug: print(f">>> Iteration: {n}")

        #
        # Extract configs from trials
        configs = get_configs(trials)

        # Get fitness and avg fitness (F_Bar)
        F = get_metrics(trials, "total_R")
        F_bar = np.mean(F)

        if debug: print(f">>> F: {F}")
        if debug: print(f">>> F_bar: {F_bar}")
        if debug: print(f">>> Current pop: {population}")

        # Replicate based on the fitness gradient
        population = (population * F) / F_bar
        population /= np.sum(population)

        if debug: print(f">>> Updated pop: {population}")

        # Cull replicators than chance
        cull = population >= (1 / population.size)
        population = population[cull]
        population /= np.sum(population)
        configs = [configs[c] for c in cull]
        if debug: print(f">>> Number surviving {np.sum(cull)}")

        # Replace the culled; Perturb to reproduce.
        num_children = int(num_replicators - population.size)
        children = []
        child_configs = []
        if debug: print(f">>> Having {num_children} children")

        for n in range(num_children):
            # Pick a random replicator ith, sampling based on its pop value
            ith = prng.choice(range(population.size), p=population)

            # Perturb ith config.
            child_config = configs[ith]
            for key, value in child_config.items():
                delta = value * perturbation
                child_config[key] = prng.uniform(value - delta, value + delta)
            child_configs.append(child_config)

            # Copy ith p.
            children.append(population[ith])

        # Update population w/ children
        population = np.concatenate([population, children])
        configs.append(child_configs)

        # Renorm after reproduction
        population /= np.sum(population)
        if debug: print(f">>> Next generation: {population}")

        # -------------------------------------------------------------------
        # Run!
        #
        # Reset trials for the next round
        trials = []

        # Re-init the pool
        workers = []
        pool = Pool(processes=num_processes)

        # Do another round of opt, with the new config
        for t in range(population.size):
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
    fire.Fire({
        "random": tune_random,
        "pbt": tune_pbt,
        "replicator": tune_replicator
    })
