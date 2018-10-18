import numpy as np


def create_distribution():
    """Create a state distribution.
    
    Note: a very thin wrapper to create an empty dict()
    """

    return {}


def value(distribution, state, p):
    """Value new information (state, p) for a discrete state distribution."""

    # TODO is this reasonable? log(1) is null....
    if len(distribution) == 0:
        return 1.0

    # Information that is known is not valuable
    if state in distribution:
        return 0.0

    # Calc info value
    n = len(distribution)
    m = n + 1
    I_intial = -np.sum([np.log(p_i) for s_i, p_i in distribution.items()])
    I_intial /= n * np.log(n)
    I_new = -np.log(p) / (m * np.log(m))
    I = I_intial - I_new

    # Update the dist.
    distribution[state] = p

    return I, distribution
