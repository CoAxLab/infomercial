import os
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from scipy.stats import loguniform

from infomercial.exp import forage

# Borrow from the other tune exp
from infomercial.exp.tune_bandit import get_best_result
from infomercial.exp.tune_bandit import get_sorted_trials
from infomercial.exp.tune_bandit import get_configs
from infomercial.exp.tune_bandit import get_metrics
from infomercial.exp.tune_bandit import save_csv

# def get_best_trial(trial_list, metric):
#     """Retrieve the best trial."""
#     return max(trial_list, key=lambda trial: trial[metric])

# def get_sorted_trials(trial_list, metric):
#     return sorted(trial_list, key=lambda trial: trial[metric], reverse=True)

# def get_best_result(trial_list, metric):
#     """Retrieve the last result from the best trial."""
#     return {metric: get_best_trial(trial_list, metric)[metric]}

# def get_configs(trial_list):
#     """Extract configs"""

#     return [trial["config"] for trial in trial_list]

# def get_metrics(trial_list, metric):
#     """Extract metric"""
#     return [trial[metric] for trial in trial_list]

# def save_csv(trials, filename=None):
#     # Init the head
#     head = list(trials[0].keys())
#     head.insert(0, 'index')

#     # Init csv writer
#     with open(filename, mode='w') as fi:
#         writer = csv.writer(fi,
#                             delimiter=',',
#                             quotechar='"',
#                             quoting=csv.QUOTE_MINIMAL)

#         # Write the header
#         writer.writerow(head)

#         # Finally, write the trials
#         for i, trial in trials.items():
#             row = [i] + list(trial.values())
#             writer.writerow(row)


def train(exp_func=None,
          metric=None,
          num_episodes=None,
          num_steps=None,
          num_repeats=None,
          master_seed=None,
          config=None):

    # Run
    scores = []
    for n in range(num_repeats):
        seed = None
        if master_seed is not None:
            seed = master_seed + n

        trial = exp_func(
            num_episodes=num_episodes,
            num_steps=num_steps,
            master_seed=seed,  # override
            **config)
        scores.append(trial[metric])

    # Override metric, with num_repeat average
    trial[metric] = np.median(scores)

    # Save metadata
    trial.update({
        "config": config,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "num_repeats": num_repeats,
        "metric": metric,
        "master_seed": master_seed,
    })

    return trial


def tune_random(name,
                exp_name='wsls',
                num_episodes=1000,
                num_steps=20,
                num_repeats=10,
                num_samples=10,
                num_processes=1,
                metric="total_R",
                log_space=False,
                master_seed=None,
                output=False,
                **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""
    prng = np.random.RandomState(master_seed)

    # ------------------------------------------------------------------------
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(forage, exp_name)

    # Build the parallel callback
    trials = []

    def append_to_results(result):
        # Keep only params and scores, to save
        # memory for when N is large
        trial = {}
        trial["config"] = result["config"]
        trial[metric] = result[metric]
        trials.append(trial)

    # Setup default params
    params = dict(exp_func=exp_func,
                  num_episodes=num_episodes,
                  num_steps=num_steps,
                  num_repeats=num_repeats,
                  metric=metric,
                  master_seed=master_seed,
                  config={})

    # ------------------------------------------------------------------------
    # Run!
    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for n in range(num_samples):

        # Reset param sample for safety
        params["config"] = {}
        params["config"]["write_to_disk"] = False
        # Make a new sample
        for k, par in config_kwargs.items():
            try:
                low, high = par
                if log_space:
                    params["config"][k] = loguniform(
                        low, high).rvs(random_state=prng)
                else:
                    params["config"][k] = prng.uniform(low=low, high=high)
            except (TypeError, ValueError):  # number or str?
                try:
                    params["config"][k] = float(par)
                except ValueError:  # string?
                    params["config"][k] = str(par)

        # A worker gets the new sample
        workers.append(
            pool.apply_async(train,
                             kwds=deepcopy(params),
                             callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in tqdm(workers):
        worker.get()
    pool.close()
    pool.join()

    # Cleanup - dump write_to_disk arg
    for trial in trials:
        del trial["config"]["write_to_disk"]

    # ------------------------------------------------------------------------
    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, metric)):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({metric: trial[metric]})
    save_csv(sorted_configs, filename=os.path.join(path, name + "_sorted.csv"))

    if output:
        return trials
    else:
        return None


if __name__ == "__main__":
    # Generate CL interface.
    import fire
    fire.Fire({"random": tune_random})
