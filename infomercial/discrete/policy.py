import numpy as np


def random(game, player, prng=None):
    if prng is None:
        prng = np.random.RandomState()

    n = game[player, :].shape[0]

    return prng.randint(0, n)


def greedy(game, player):
    return np.argmax(game.detach().numpy(), axis=player)
