import numpy as np
from itertools import product
from scipy.stats import entropy as scientropy
from collections import OrderedDict


def lp(x, y, p):
    x = np.asarray(x)
    y = np.asarray(y)
    deltas = np.power(np.abs(x - y), p)
    return np.power(np.sum(deltas), 1 / p)


def l2(x, y):
    return lp(x, y, 2)


def l1(x, y):
    return lp(x, y, 1)


def linf(x, y):
    return np.max(np.abs(x - y))


def kl(p, q, base=None):
    return scientropy(p, q, base=base)
