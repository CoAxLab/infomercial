import numpy as np


def create_distribution():
    """Create a state distribution.
    
    Note: a very thin wrapper to create an empty dict()
    """

    return {}


def update_distribution(distribution, state, p):
    # Base case
    if state in distribution:
        if np.isclose(distribution[state], p):
            return distribution

    # Update
    distribution[state] = p

    # Renorm
    # sum_p = np.sum(list(distribution.values()))
    # for k, v in distribution.items():
    # distribution[k] /= sum_p

    return distribution


def information_value(distribution, state, p, n):
    """Value new information (state, p) for a discrete state distribution."""

    # Max surprise
    I_max = (n * np.log2(n))

    # Current suprise
    I_intial = -np.sum([np.log2(p_i) for s_i, p_i in distribution.items()])
    I_intial /= I_max

    # Update dist?
    distribution = update_distribution(distribution, state, p)

    # New surprise
    I_new = -np.sum([np.log2(p_i) for s_i, p_i in distribution.items()])
    I_new /= I_max

    I = I_new - I_intial

    return I, distribution
