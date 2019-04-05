import fire
import ray
import os
from ray.tune import sample_from
from ray.tune import run as ray_run
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining

import numpy as np
from infomercial.exp import meta_bandit
from infomercial.exp import beta_bandit
from infomercial.exp import epsilon_bandit
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


class TuneBase(Trainable):
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


class TuneBeta(TuneBase):
    def _train(self):
        self.result = beta_bandit(**self.config)
        self.iteration += 1
        return self.result


class TuneMeta(TuneBase):
    def _train(self):
        self.result = meta_bandit(**self.config)
        self.iteration += 1
        return self.result


class TuneEpsilon(TuneBase):
    def _train(self):
        self.result = epsilon_bandit(**self.config)
        self.iteration += 1
        return self.result


def run(name, exp_name='beta_bandit', num_samples=10, **config_kwargs):
    """Tune hyperparameters of any bandit experiment."""

    # Separate name from path
    path, name = os.path.split(name)

    # Build the config dict
    config = {}  # Home for processed kwargs
    keys = sorted(list(config_kwargs.keys()))
    for k in keys:
        # Either it is a param list, so make it a sampling fn
        try:
            begin, end = config_kwargs[k]
            v = lambda spec: np.random.uniform(begin, end)
            v = sample_from(v)
        # Or a scalar
        except TypeError:
            v = config_kwargs[k]
        except ValueError:
            v = config_kwargs[k]

        config[k] = v

    if exp_name == 'beta_bandit':
        Tuner = TuneBeta
    elif exp_name == 'epsilon_bandit':
        Tuner = TuneEpsilon
    elif exp_name == 'meta_bandit':
        Tuner = TuneMeta
    else:
        raise ValueError("exp_name not known")

    pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        reward_attr='total_R',
        perturbation_interval=600.0,
        hyperparam_mutations=config)

    stop = {"training_iteration": 20}
    trials = ray_run(
        Tuner,
        name=name,
        local_dir=path,
        num_samples=num_samples,
        reuse_actors=True,
        stop=stop,
        config=config,
        scheduler=pbt,
        verbose=False)
    best = get_best_trial(trials, 'total_R')

    # Save best params in an easy to find place
    best_config = best.config
    best_config.update(get_best_result(trials, 'total_R'))
    save_checkpoint(
        best_config, filename=os.path.join(path, name + "_best.pkl"))

    # Sum all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
        sorted_configs[i] = trial.config
        sorted_configs[i].update({"total_R": trial.last_result["total_R"]})
    save_checkpoint(
        sorted_configs, filename=os.path.join(path, name + "_sorted.pkl"))

    # -
    return best, trials


if __name__ == "__main__":
    fire.Fire(run)
