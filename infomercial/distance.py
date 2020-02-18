import numpy as np
from itertools import product
from scipy.stats import entropy as scientropy
from collections import OrderedDict


def _lp(x, y, p):
    x = np.asarray(x)
    y = np.asarray(y)
    deltas = np.power(np.abs(x - y), p)
    return np.power(np.sum(deltas), 1 / p)


def l2(memory_new, memory_old, default):
    if len(memory_old) == 0:
        return default
    if len(memory_new) == 0:
        return default

    # Find a common set of keys
    keys = set(memory_new.keys() + memory_old.keys())

    # Get values
    x = [memory_old(k) for k in keys]
    y = [memory_new(k) for k in keys]

    return _lp(x, y, 2)


def l1(memory_new, memory_old, default):
    if len(memory_old) == 0:
        return default
    if len(memory_new) == 0:
        return default

    # Find a common set of keys
    keys = set(memory_new.keys() + memory_old.keys())

    # Get values
    x = [memory_old(k) for k in keys]
    y = [memory_new(k) for k in keys]

    return _lp(x, y, 1)


def linf(memory_new, memory_old, default):
    if len(memory_old) == 0:
        return default
    if len(memory_new) == 0:
        return default

    # Find a common set of keys
    keys = set(memory_new.keys() + memory_old.keys())

    # Get values
    x = [memory_old(k) for k in keys]
    y = [memory_new(k) for k in keys]

    return np.max(np.abs(x - y))


def kl(memory_new, memory_old, default, base=None):
    """Calculate KL, assuming prob. memoroes."""
    if len(memory_old) == 0:
        return default
    if len(memory_new) == 0:
        return default

    # Find a common set of keys
    keys = set(memory_new.keys() + memory_old.keys())

    # Get ps
    p_old = [memory_old(k) for k in keys]
    p_new = [memory_new(k) for k in keys]

    if np.isclose(np.sum(p_old), 0):
        return default
    if np.isclose(np.sum(p_new), 0):
        return default

    return scientropy(p_old, qk=p_new, base=base)
