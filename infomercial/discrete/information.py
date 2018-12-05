import numpy as np
from scipy.stats import entropy


def create_distribution():
    """Create a state distribution.
    
    Note: a very thin wrapper to create an empty dict()
    """

    return {}


def update_distribution(distribution, state, p, renorm=True):
    # Base case
    if state in distribution:
        if np.isclose(distribution[state], p):
            return distribution

    # Update
    distribution[state] = p

    # Renorm
    if renorm:
        sum_p = np.sum(list(distribution.values()))
        for k, v in distribution.items():
            distribution[k] /= sum_p

    return distribution


def get_probs(distribution):
    return np.asarray(list(distribution.values()))


def get_diff(probs1, probs2):
    """Get the total difference in probability"""
    probs1 = np.asarray(probs1)
    probs2 = np.asarray(probs2)
    l1 = probs1.size
    l2 = probs2.size

    if l1 == l2:
        d = probs2 - probs1
    else:
        minl = min(l1, l2)
        d = probs2[0:minl] - probs1[0:minl]

        if l2 > l1:
            d = np.concatenate(d, probs2[l1:])
        else:
            d = np.concatenate(d, probs1[l2:])

    return np.sum(d)


def information_value(distribution, state, p):
    """Value new information (state, p) for a discrete state distribution."""

    # Initial H
    probs = get_probs(distribution)
    H = entropy(probs)

    # New H'
    distribution = update_distribution(distribution, state, p)
    probs_prime = get_probs(distribution)
    H_prime = entropy(probs_prime)

    # Derivatives
    dH = H_prime - H
    dX = get_diff(probs, probs_prime)

    # Info value
    M = np.abs(dH / dX)

    return M, distribution
