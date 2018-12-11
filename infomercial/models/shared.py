import numpy as npnp
import numpy.random as npr
import jax.numpy as np

EPS = npnp.finfo(npnp.float32).eps.item()


def relu(x):
    return np.max(0, x)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 0.5 * (tanh(x / 2.) + 1)


def softmax(x, axis=1, keepdims=True):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=keepdims)


def init_dense_params(scale, layer_sizes, rng=npr.RandomState(0)):
    if layer_sizes is None:
        raise ValueError("layer_sizes can't be the default None.")

    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


def linear_policy(params, inputs):
    for w, b in params:
        outputs = np.dot(inputs, w) + b
        inputs = outputs  # shuffle
    return softmax(outputs)


def nonlinear_policy(params, inputs, activation_function=tanh):
    for w, b in params:
        outputs = np.dot(inputs, w) + b
        inputs = activation_function(outputs)  # nonlin then shuffle
    return softmax(outputs)


def step(params, env, inputs, policy=linear_policy, action_space='discrete'):
    """Take a step in a Gym env."""
    inputs = np.asarray(inputs)

    probs = policy(params, inputs)
    if action_space == 'discrete':
        action = npnp.random.choice(range(len(probs)), p=npnp.asarray(probs))
    else:
        raise NotImplementedError("Continuous modes: TODO.")
    outputs, reward, done, _ = env.step(action)

    return env, action, probs, outputs, reward, done
