import fire
import ray
import os
from copy import deepcopy

from functools import partial
from multiprocessing import Pool

import numpy as np

from infomercial import exp
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint

# from ray.tune import sample_from
# from ray.tune import function
# from ray.tune import grid_search
# from ray.tune import function
# from ray.tune import run as ray_run
# from ray.tune import Trainable
# from ray.tune.schedulers import PopulationBasedTraining


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
    trial = exp_func(env_name=env_name,
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
                metric="total_R",
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
    params = dict(exp_func=exp_func,
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
            pool.apply_async(train,
                             kwds=deepcopy(params),
                             callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in workers:
        worker.get()
    pool.close()
    pool.join()

    # ------------------------------------------------------------------------
    # Save configs
    best = get_best_trial(trials, metric)

    # Best trial config
    best_config = best["config"]
    best_config.update(get_best_result(trials, metric))
    save_checkpoint(best_config,
                    filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, metric)):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({metric: trial[metric]})
    save_checkpoint(sorted_configs,
                    filename=os.path.join(path, name + "_sorted.pkl"))

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
    params = dict(exp_func=exp_func,
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
            pool.apply_async(train,
                             kwds=deepcopy(params),
                             callback=append_to_results))

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
            params["num_episodes"] += int(params["num_episodes"] *
                                          top_threshold)
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
                pool.apply_async(train,
                                 kwds=deepcopy(params),
                                 callback=append_to_results))

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
    save_checkpoint(best_config,
                    filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({"total_R": trial["total_R"]})
    save_checkpoint(sorted_configs,
                    filename=os.path.join(path, name + "_sorted.pkl"))

    return best, trials


def tune_replicator(name,
                    exp_name='beta_bandit',
                    env_name='BanditOneHigh10-v0',
                    num_iterations=2,
                    num_episodes=2000,
                    num_replicators=10,
                    num_processes=1,
                    perturbation=0.1,
                    metric="total_R",
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
    params = dict(exp_func=exp_func,
                  env_name=env_name,
                  num_episodes=num_episodes,
                  seed_value=seed_value,
                  config={})

    # ------------------------------------------------------------------------
    # Run first population!
    population = np.ones(num_replicators) / num_replicators

    if verbose: print(f">>> Intial population: {population}")

    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for t in range(num_replicators):

        # Reset param sample for safety
        params["config"] = {}

        # Make a new sample
        for k, config_kwarg in config_kwargs.items():
            try:
                low, high = config_kwarg
                params["config"][k] = prng.uniform(low=low, high=high)
            except TypeError:
                params["config"][k] = config_kwarg

        # A worker gets the new sample
        workers.append(
            pool.apply_async(train,
                             kwds=deepcopy(params),
                             callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in workers:
        worker.get()
    pool.close()
    pool.join()
    pool.terminate()

    if verbose: print(f">>> Example intial config{params['config']}")

    # ------------------------------------------------------------------------
    # Init the meta-population of perturbation strategies
    meta = False
    if perturbation == 'meta':
        meta = True
    if meta:
        perturbations = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        meta_population = np.ones(len(perturbations)) / len(perturbations)
        F_meta = np.ones_like(meta_population) * np.mean(
            get_metrics(trials, metric))

        if verbose: print(f"\n>>> Perturbations: {perturbations}")
        if verbose: print(f">>> Initial F_meta: {F_meta}")

    # ------------------------------------------------------------------------
    # Optimize for num_iterations
    for n in range(num_iterations):
        # --------------------------------------------------------------------
        # Replicate!
        if verbose: print(f"\n>>> ---Begining new replicatation!---")
        if verbose: print(f">>> Iteration: {n}")

        #
        # Extract configs from trials
        configs = get_configs(trials)

        # Get fitness and avg fitness (F_Bar)
        F = get_metrics(trials, metric)
        F_bar = np.mean(F)

        if verbose: print(f">>> F: {F}")
        if verbose: print(f">>> F_bar: {F_bar}")
        if verbose: print(f">>> Current pop: {population}")

        # Replicate based on the fitness gradient
        population = (population * F) / F_bar
        population /= np.sum(population)

        if verbose: print(f">>> Updated pop: {population}")
        if verbose: print(f">>> Pop size: {population.size}")

        # Cull replicators less than chance
        cull = population >= (1 / population.size)
        population = population[cull]
        population /= np.sum(population)
        configs = [c for (c, m) in zip(configs, cull) if m]

        if verbose:
            print(f">>> Number surviving: {np.sum(cull)}")

        # Replace the culled; Perturb to reproduce.
        num_children = int(num_replicators - population.size)
        children = []
        child_configs = []

        if verbose: print(f">>> Having {num_children} children")

        # Sample from the meta_population to find a perturbation
        if meta:
            ith_meta = prng.choice(range(meta_population.size),
                                   p=meta_population)
            perturbation = perturbations[ith_meta]

            if verbose: print(f">>> perturbation {perturbation} ({ith_meta})")

        # Have kids, by perturbation.
        for n in range(num_children):
            # Pick a random replicator ith, sampling based on its pop value
            ith = prng.choice(range(population.size), p=population)

            # Perturb ith config.
            child_config = deepcopy(configs[ith])
            for key, value in child_config.items():
                delta = value * perturbation
                xi = prng.uniform(value - delta, value + delta)
                child_config[key] = xi

            if verbose:
                print(f">>> Set {ith} chosen ({configs[ith]})")
                print(f">>> New child: {child_config}" "")

            # Copy ith p.
            child_configs.append(child_config)
            children.append(population[ith])

        assert len(child_configs) == num_children, "Reproduction error."
        if verbose: print(f"child configs: {child_configs}")

        # Update population w/ children
        population = np.concatenate([population, children])
        configs.extend(child_configs)

        # Renorm after reproduction
        population /= np.sum(population)

        assert len(configs) == population.size, "Regrouping error!"
        if verbose: print(f">>> configs: {configs}")
        if verbose: print(f">>> Next generation: {population}")

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
                pool.apply_async(train,
                                 kwds=deepcopy(params),
                                 callback=append_to_results))

        # Get the worker's result (blocks until complete)
        for worker in workers:
            worker.get()
        pool.close()
        pool.join()
        pool.terminate()

        # -------------------------------------------------------------------
        # Update the meta-population.
        if meta:
            # Get fitness
            F_meta[ith_meta] = np.mean(get_metrics(trials, metric))
            F_bar_meta = np.mean(F_meta)

            if verbose: print(f">>> meta F: {F_meta}")
            if verbose: print(f">>> meta F_bar: {F_bar_meta}")
            if verbose: print(f">>> meta current pop: {meta_population}")

            # Meta-replicate, still based on the fitness gradient but
            # only update the ith perturbation used in last iteration.
            p_ith = meta_population[ith_meta]
            F_ith = F_meta[ith_meta]
            meta_population[ith_meta] = (p_ith * F_ith) / F_bar_meta
            meta_population /= np.sum(meta_population)

            if verbose: print(f">>> Updated meta pop: {meta_population}")

    # ------------------------------------------------------------------------
    # Save configs and metric (full model data is dropped):
    best = get_best_trial(trials, metric)

    # Best trial config
    best_config = best["config"]
    best_config.update(get_best_result(trials, metric))
    save_checkpoint(best_config,
                    filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, metric)):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({metric: trial[metric]})
    save_checkpoint(sorted_configs,
                    filename=os.path.join(path, name + "_sorted.pkl"))

    return best, trials


if __name__ == "__main__":
    # Generate CL interface.
    fire.Fire({
        "random": tune_random,
        "pbt": tune_pbt,
        "replicator": tune_replicator
    })
