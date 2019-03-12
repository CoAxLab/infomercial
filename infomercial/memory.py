import jax.numpy as np
from jax import random
import numpy as onp
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Discrete distributions/memories
def init_count_params():
    return 0, OrderedDict()


def sample_count(params, size=None):
    N, counts = params

    X = list(counts.keys())
    p = onp.asarray[prob_count(params, x) for x in X]

    return onp.random.choice(X, size=size, p=p)


def update_count(params, x):
    # Unpack
    N, counts = params

    # Update
    if x in counts:
        counts[x] += 1
    else:
        counts[x] = 1
    N += 1

    # Repack
    params = N, counts

    return params


def prob_count(params, x):
    N, counts = params
    return counts[x] / N


# ---------------------------------------------------------------------------
# Kernel memory for continous data

# ---------------------------------------------------------------------------
# MADE memory high-dimensional data

# ---------------------------------------------------------------------------
# beta-MADE memory high-dimensional data
