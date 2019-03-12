import numpy as np
from itertools import product
from scipy.stats import entropy as scientropy
from collections import OrderedDict


def estimate_prob(X):
    """Estimate the prob of each unique element in X"""

    X = np.asarray(X)
    if X.ndim != 1:
        raise ValueError("X must be 1d.")
    if X.size == 0:
        return 0.0

    # Est P(.) for each symbol in X
    probs = []
    for c1 in set(X):
        probs.append(np.mean(c1 == X))
    probs = np.asarray(probs)

    probs = probs[probs > 0]

    return probs, list(set(X))


def cond_probs(X, Y):
    """Estimate the prob of each unique element in (X, Y)"""

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 1:
        raise ValueError("X must be 1d.")
    if Y.ndim != 1:
        raise ValueError("Y must be 1d.")
    if X.size == 0:
        return 0.0
    if Y.size == 0:
        return 0.0

    # Est P(.) for each symbol in X
    probs = []
    conds = list(product(set(X), set(Y)))
    for c1, c2 in conds:
        probs.append(np.mean(np.logical_and(X == c1, Y == c2)))
    probs = np.asarray(probs)

    if np.isnan(probs).sum() > 0:
        print(probs)
        raise ValueError("p est if very off")

    probs = probs[probs > 0]

    return probs, conds


def entropy(X):
    """Entropy for a list of symbols, X."""

    probs, _ = estimate_prob(X)
    return -np.sum(probs * np.log(probs))


def cond_entropy(X, Y):
    """Conditional entropy for lists of symbols, X and Y."""

    probs, _ = cond_probs(X, Y)
    return -np.sum(probs * np.log(probs))


def surprisal(X, Y):
    """Estimate the difference in surprise between X and Y"""
    X = np.asarray(X)
    Y = np.asarray(Y)

    prob1 = estimate_prob(X)[0]
    prob2 = estimate_prob(Y)[0]

    s1 = np.sum(np.log(1 / prob1))
    s2 = np.sum(np.log(1 / prob2))

    return np.abs(s2 - s1)


def mutual_information(X, Y):
    """Discrete mutual information (no bias correction)
    Note: Only supports 1d inputs, and integer values.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    return entropy(X) + entropy(Y) - cond_entropy(X, Y)


def kl_divergence(a, b):
    """Calculate the K-L divergence between a and b
    
    Note: a and b must be two sequences of integers
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Find the total set of symbols
    a_set = set(a)
    b_set = set(b)
    ab_set = a_set.union(b_set)

    # Create a lookup table for each symbol in p_a/p_b
    lookup = {}
    for i, x in enumerate(ab_set):
        lookup[x] = i

    # Calculate event probabilities for and then b
    # To prevent nan/division errors every event
    # gets at least a 1 count.
    p_a = np.ones(len(ab_set))
    for x in a:
        p_a[lookup[x]] += 1

    p_b = np.ones(len(ab_set))
    for x in b:
        p_b[lookup[x]] += 1

    # Norm counts into probabilities
    p_a /= a.size
    p_b /= b.size

    return scientropy(p_a, p_b)


def janson_shannon(a, b):
    """The Janson-Shannon divergence"""

    a = np.asarray(a)
    b = np.asarray(b)

    # Find the total set of symbols
    a_set = set(a)
    b_set = set(b)
    ab_set = a_set.union(b_set)

    # Create a lookup table for each symbol in p_a/p_b
    lookup = {}
    for i, x in enumerate(ab_set):
        lookup[x] = i

    # Calculate event probabilities for and then b
    # To prevent nan/division errors every event
    # gets at least a 1 count.
    p_a = np.ones(len(ab_set))
    for x in a:
        p_a[lookup[x]] += 1

    p_b = np.ones(len(ab_set))
    for x in b:
        p_b[lookup[x]] += 1

    # Norm counts into probabilities
    p_a /= a.size
    p_b /= b.size
    m = 0.5 * (p_a + p_b)

    return scientropy(p_a, m) / 2 + scientropy(p_b, m) / 2.


def delta_p(X, Y):
    """Get the total difference in probability"""
    prob1 = estimate_prob(X)[0]
    prob2 = estimate_prob(Y)[0]

    return np.sum(np.abs(prob2 - prob1)) / 2


def delta_H(X, Y):
    """Value information between two prob vectors."""
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Est. p(X), H(X), ...
    H1 = entropy(X)
    H2 = entropy(Y)

    # Est. info value
    dH = H2 - H1
    dX = delta_p(X, Y)
    if np.isclose(dX, 0):
        E = 0.0
    else:
        E = np.abs(dH / dX)

    return E
