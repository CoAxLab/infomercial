import numpy as np
from scipy.stats import entropy
from collections import OrderedDict


def _get_diff(prob1, prob2):
    """Get the total difference in probability"""
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    l1 = prob1.size
    l2 = prob2.size

    if l1 == l2:
        d = prob2 - prob1
    else:
        minl = min(l1, l2)
        d = prob2[0:minl] - prob1[0:minl]

        if l2 > l1:
            d = np.concatenate(d, prob2[l1:])
        else:
            d = np.concatenate(d, prob1[l2:])

    return np.sum(d)


def information_value(prob1, prob2):
    """Value information between two prob vaectors."""
    # Est H
    H1 = entropy(prob1)
    H2 = entropy(prob2)

    # Derivatives
    dH = H2 - H1
    dX = _get_diff(prob1, prob2)

    # Info value
    M = np.abs(dH / dX)

    return M
