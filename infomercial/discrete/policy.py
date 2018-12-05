import numpy as np


def random(x, prng=None):
    if prng is None:
        prng = np.random.RandomState()
    i = prng.randint(0, len(x))
    return i, x[i]


def greedy(x):
    i = np.argmax(x)
    return i, x[i]


def anti_greedy(x):
    i = np.argmin(x)
    return i, x[i]
