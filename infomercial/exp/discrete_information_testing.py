#%%
import numpy as np
from infomercial.discrete.games import uniform
from infomercial.discrete.games import irregular
from infomercial.discrete.games import prisoners_dilemma
from infomercial.discrete.games import create_state
from infomercial.discrete import information_value
from infomercial.discrete import create_distribution
from infomercial.discrete.policy import random
from infomercial.discrete.policy import greedy

prng = np.random.RandomState(42)

#%%
n_states = 20
n_players = 2
game = uniform(n_players, n_states)
game = irregular(n_players, n_states)
# states = create_state(game)

# print(">>> The game")
# print(game)

#%%
dist1 = create_distribution()
dist2 = create_distribution()
values1 = []
values2 = []

max_iterations = 50
for i in range(max_iterations):
    # p1
    move = random(game, 0, prng=prng)
    new = False
    if move not in dist1:
        new = True

    p = game[move]
    value, dist1 = information_value(dist1, move, p, n_states)
    if new or not new:
        print(">>> P1: iter {}, move {}, value {}, new {}".format(
            i, move, value, new))

    values1.append(value)

    # p2
    move = random(game, 0, prng=prng)
    p = game[move]
    value, dist2 = information_value(dist2, move, p, n_states)
    values2.append(value)

print(">>> Value series (p1, p2)")
print(values1)
print(values2)
