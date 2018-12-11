import numpy as np

# TODO make API gym like


def uniform(n_players, n_states):
    """Create an equal information game"""
    game = np.ones((n_players, n_states, n_states)) / n_states

    return game


def irregular(n_players, n_states):
    game = np.random.rand(n_players, n_states, n_states)
    game /= np.sum(game)

    return game


def prisoners_dilemma():
    """Create a Prisoners dilemma."""

    payout = np.empty((2, 4))
    payout[0, :] = [[-1, 1], [1, -1]]
    payout[1, :] = [[1, -1], [-1, 1]]

    return payout


def create_state(game):
    """Use (i, j, k, ...) indices to define states for any matrix game.
    
    Note: Assumes i is always the n_player index.
    """

    return list(np.ndindex(*game.shape))
