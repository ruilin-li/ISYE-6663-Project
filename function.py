import numpy as np


# extender_rosenbrock_function
def extended_rosenbrock_function(x):
    """
    Compute the value of extended rosenbrock function at point x.
    """
    odd = x[::2]
    even = x[1::2]

    return 100 * np.square(even - np.square(odd)).sum() \
        + np.square(1 - odd).sum()


# gradient of extender_rosenbrock_function
def gradient_erf(x):
    """
    Compute the gradient of extended rosenbrock function at point x.

    Return: np.array
    """
    odd = x[0::2]
    even = x[1::2]

    grad = np.zeros(x.shape)
    grad[0::2] = 400 * np.multiply(np.square(odd) - even, odd) + 2 * (odd - 1)
    grad[1::2] = 200 * (even - np.square(odd))

    return grad


def phi_erf(x, p , alpha):
    return extended_rosenbrock_function(x + alpha * p)


def grad_phi_erf(x, p, alpha):
    return np.dot(phi_erf(x, p, alpha), p)


# extended_powell_singular_function
def extended_powell_singular_function(x):
    """Compute the value of extended powell singular function at point x."""
    x1 = x[0::4]
    x2 = x[1::4]
    x3 = x[2::4]
    x4 = x[3::4]

    return np.square(x1 + 10 * x2).sum() \
        + 5 * np.square(x3 - x4).sum() \
        + np.power(x2 - 2 * np.square(x3), 4).sum() \
        + 10 * np.power(x1 - x4, 4).sum()


# gradient of extender_powell_singular_function
def gradient_epsf(x):
    """
    Compute the gradient of extended powell singular function at point x.

    Return: np.array
    """
    x1 = x[0::4]
    x2 = x[1::4]
    x3 = x[2::4]
    x4 = x[3::4]

    grad = np.zeros(x.shape)
    grad[0::4] = 2 * (x1 + 10 * x2) + 40 * np.power(x1 - x4, 3)
    grad[1::4] = 20 * (10 * x2 + x1) + 4 * np.power(x2 - 2 * np.square(x3), 3)
    grad[2::4] = 10 * (x3 - x4) + 16 * np.multiply(np.power(2 * np.square(x3) - x2, 3), x3)
    grad[3::4] = 10 * (x4 - x3) + 40 * np.power(x4 - x1, 3)

    return grad


def phi_epsf(x, p , alpha):
    return extended_powell_singular_function(x + alpha * p)


def grad_phi_epsf(x, p, alpha):
    return np.dot(phi_epsf(x, p, alpha), p)
