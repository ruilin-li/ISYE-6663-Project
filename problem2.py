import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import ad

dimension = 1000
A = scipy.sparse.rand(5, dimension).todense()
b = np.ones(5)

# beta convexity
beta = 2 * np.max(np.square(A).sum())


def f(x):
    """Compute the value of function f."""
    if x.shape == (1000, 1):
        x = x.T
    return np.linalg.norm(A.dot(x) - b)


def grad_f(x):
    """"Compute the gradient of function f."""
    return np.asarray(2 * (A.dot(x) - b).dot(A)).reshape(-1)


def nesterov_1(init_x, eps=1e-6):
    """Use Nesterov accelarated method to solve the unconstrained problem."""
    init_f = f(init_x)

    lam = 1
    x = np.copy(init_x)
    y = np.copy(init_x)
    current_f = f(y)

    iteration = 0

    while current_f / init_f > eps:
        # if iteration < 10:
        # print "iteration: {} function value: {}".format(iteration, f(x))
        # print "iteration: {} grad: {}".format(iteration, np.linalg.norm(grad_f(x)))

        next_y = x - (1 / beta) * grad_f(x)
        next_lam = (1 + np.sqrt(1 + 4 * lam**2)) / 2
        gamma = (1 - lam) / next_lam
        next_x = (1 - gamma) * next_y + gamma * y

        lam = next_lam
        x = next_x
        y = next_y
        current_f = f(y)

        iteration += 1

    return (y, current_f, iteration)

def project(x):
    """Project x to feasible region."""
    x[x<0] = 0
    return x

def nesterov_2(init_x, eps=1e-6):
    """Use Nesterov accelarated method to solve the constrained problem."""
    init_f = f(init_x)

    lam = 1
    y = project(np.copy(init_x))
    next_y = project(y - 1 / beta * grad_f(y))
    current_f = f(y)

    iteration = 0

    while np.linalg.norm(next_y - y) > eps:
        # print "iteration: {} function value: {}".format(iteration, f(y))
        # print "iteration: {} grad: {}".format(iteration, np.linalg.norm(grad_f(y)))

        next_lam = (1 + np.sqrt(1 + 4 * lam**2)) / 2
        gamma = (1 - lam) / next_lam
        x = (1 - gamma) * next_y + gamma * y

        y = next_y
        next_y = project(x - 1 / beta * grad_f(x))
        lam = next_lam
        current_f = f(y)

        iteration += 1

    return (y, current_f, iteration)

if __name__ == "__main__":
    x0 = np.zeros(dimension)
    # nesterov_1(x0)
    # nesterov_2(x0)
