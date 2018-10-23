import numpy as np


def random(game, player, prng=None):
    if prng is None:
        prng = np.random.RandomState()

    n = game[player, :].shape[0]
    i = prng.randint(0, n)
    return (player, i, i)


def greedy(game, player):
    i = np.argmax(game[player, ...], axis=player)

    return (player, i, i)
