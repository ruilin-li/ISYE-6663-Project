import numpy as np
import time
from function import *
from scipy.optimize import line_search
from memory_profiler import profile


class Optimizer(object):
    """An optimizizer class."""

    def __init__(self, f):
        """
        Initialize an optimizer object.

        Parameters:
        ----------
            f: objective function to be minimized, 'erf' or 'epsf'
        """
        if f == 'erf':
            self.f = extended_rosenbrock_function
            self.grad_f = gradient_erf
            self.phi = phi_erf
            self.grad_phi = grad_phi_erf
        elif f == 'epsf':
            self.f = extended_powell_singular_function
            self.grad_f = gradient_epsf
            self.phi = phi_epsf
            self.grad_phi = grad_phi_epsf
        else:
            raise ValueError("The input function is not supported.")

    def check_dimension(self, init_x):
        """Check the dimension of input."""
        if self.f == extended_rosenbrock_function and init_x.size % 2 != 0:
            raise ValueError("Invalid input dimension.")
        elif self.f == extended_powell_singular_function and init_x.size % 4 != 0:
            raise ValueError("Invalide input dimension")
    
    @profile
    def steepest_descent(self, init_x, alpha=0.002, eps=1e-6, store=True):
        """
        Use steepest descent to minimize a objective function.

        Parameters:
        -----------
            init_x: initial start point
            step: step size
            eps: termination tolerance
            store: whether to store the optimization process

        Return: (current_x, current_f, iteration, hist_x, hist_f, lag)
        -------
            current_x: the minimizer found by the algorithm
            curremt_f: the minimum found by t he algorithm
            iteration: total iterations
            hist_x: positions of x during optimization process
            hist_f: values of objective function during optimization process
            lag: the lag of iterations between two consecutive points in hist_x
        """
        self.check_dimension(init_x)

        init_f = self.f(init_x)
        x, f, g = np.copy(init_x), init_f, self.grad_f(init_x)

        hist_x = [x]
        hist_f = [f]

        iteration = 0
        lag = 100

        while f / init_f > eps:
            if iteration % lag == 0:
                if store:
                    hist_x.append(x)
                    hist_f.append(f)
                # print "iteration {}: {}".format(iteration, f)

            alpha = line_search(self.f, self.grad_f, x, -g)[0]
            x += alpha * (-g)
            f, g = self.f(x), self.grad_f(x)

            iteration += 1

        return (x, f, iteration,  hist_x, hist_f, lag)

    def line_search_wolfe(self, current_x, current_p, alpha=1, beta=0.95, c1=1e-4, c2=0.9,
                        max_iteration=200):
        """
        Find the step size satisfying Wolfe condition.

        Parameters:
        -----------
            current_x: current positions
            current_p: current search direction=
            alpha: initial step size
            beta: decaying coefficient
            c1: the constant in Armijo's condition
            c2: the constant in curvature condition

        Return:
        -------
            alpha: the desired step size
        """
        terminate = False
        alpha = alpha
        iteration = 0

        current_f = self.f(current_x)
        current_g = self.grad_f(current_x)

        while terminate is False:
            next_x = current_x + alpha * current_p
            next_f = self.f(next_x)
            next_g = self.grad_f(next_x)

            if next_f <= current_f + c1 * alpha * np.dot(current_g, current_p) \
            and np.abs(np.dot(next_g, current_p)) <= -c2 * np.dot(current_g, current_p) \
            or iteration == max_iteration:
                terminate = True
            else:
                alpha = alpha * beta

            iteration += 1

        return alpha

    
    @profile
    def conjugate_gradient_fr(self, init_x, eps=1e-6, store=False, scipy_ls=True):
        """
        Use conjugate gradient Fletcher-Reeves variant to minimize the objective function.

        Parameters:
        -----------
            init_x: initial start point
            eps: termination tolerance
            store: whether to store the optimization process

        Return: (current_x, current_f, iteration, hist_x, hist_f, lag)
        -------
            current_x: the minimizer found by the algorithm
            curremt_f: the minimum found by t he algorithm
            iteration: total iterations
            hist_x: positions of x during optimization process
            hist_f: values of objective function during optimization process
            lag: the lag of iterations between two consecutive points in hist_x
        """
        self.check_dimension(init_x)

        init_f, init_g, init_p = self.f(init_x), self.grad_f(init_x), -self.grad_f(init_x)
        current_x, current_f, current_g, current_p = np.copy(init_x), init_f, init_g, init_p

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = 0
        lag = 1

        while current_f / init_f > eps:
            current_f = self.f(current_x)

            if scipy_ls:
                alpha = line_search(self.f, self.grad_f, current_x, current_p, c2=0.05)[0]
            else:
                alpha = self.line_search_wolfe(current_x, current_p, alpha=1, beta=0.95, c1=1e-4, c2=0.9)


            next_x = current_x + alpha * current_p
            next_g = self.grad_f(next_x)
            beta = np.dot(next_g, next_g) / np.dot(current_g, current_g)
            next_p = - next_g + beta * current_p

            if iteration % lag == 0:
                if store:
                    hist_x.append(current_x)
                    hist_f.append(current_f)
                # print "iteration: {}: {}".format(iteration, current_f)

            iteration += 1

            current_x, current_g, current_p = next_x, next_g, next_p

        return (current_x, current_f, iteration,  hist_x, hist_f, lag)

    @profile
    def conjugate_gradient_pr(self, init_x, eps=1e-6, store=False):
        """
        Use conjugate gradient Polak-Ribiere variant to minimize the objective function.

        Parameters:
        -----------
            init_x: initial start point
            eps: termination tolerance\
            store: whether to store the optimization process

        Return: (current_x, current_f, iteration, hist_x, hist_f, lag)
        -------
            current_x: the minimizer found by the algorithm
            curremt_f: the minimum found by t he algorithm
            iteration: total iterations
            hist_x: positions of x during optimization process
            hist_f: values of objective function during optimization process
            lag: the lag of iterations between two consecutive points in hist_x
        """
        self.check_dimension(init_x)

        init_f, init_g, init_p = self.f(init_x), self.grad_f(init_x), -self.grad_f(init_x)
        current_x, current_f, current_g, current_p = np.copy(init_x), init_f, init_g, init_p

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = 0
        lag = 100

        while current_f / init_f > eps:

            alpha = line_search(self.f, self.grad_f, current_x, current_p, c2=0.1)[0]

            next_x = current_x + alpha * current_p
            next_g = self.grad_f(next_x)
            beta = np.dot(next_g, next_g - current_g) / np.dot(current_g, current_g)
            if beta < 0:
                beta = 0

            next_p = - next_g + beta * current_p

            if iteration % lag == 0:
                if store:
                    hist_x.append(current_x)
                    hist_f.append(current_f)
                # print "iteration {}: value: {}".format(iteration, current_f)
                # print "iteration {}: alpha: {}".format(iteration, alpha)
                # print "iteration {}: p-norm: {}".format(iteration, np.linalg.norm(next_p))

            iteration += 1

            current_x, current_f, current_g, current_p = next_x, self.f(next_x), next_g, next_p

        return (current_x, current_f, iteration,  hist_x, hist_f, lag)
    
    @profile
    def quasi_newton_bfgs(self, init_x, eps=1e-6, store=False):
        self.check_dimension(init_x)
        size = init_x.size

        init_f = self.f(init_x)
        current_x, current_f, current_g, current_H = np.copy(init_x), \
            init_f, self.grad_f(init_x), np.eye(size)

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = 0
        lag = 100

        while current_f / init_f > eps:
            current_p = - np.dot(current_H, current_g)

            alpha = line_search(self.f, self.grad_f, current_x, current_p)[0]
            # alpha = self.line_search_wolfe(current_x, current_p)
            next_x = current_x + alpha * current_p
            next_f = self.f(next_x)
            next_g = self.grad_f(next_x)
            s = alpha * current_p
            y = next_g - current_g

            rho = np.dot(s, y)
            Hy = current_H.dot(y)

            next_H = current_H \
                    + (rho + Hy.dot(y)) * np.outer(s, s) / rho**2 \
                    - (np.outer(Hy, s) + np.outer(s, Hy)) / rho

            if iteration % lag == 0:
                if store:
                    hist_x.append(current_x)
                    hist_f.append(current_f)
                # print "iteration {}: {}".format(iteration, current_f)

            iteration += 1

            current_x = next_x
            current_f = next_f
            current_g = next_g
            current_H = next_H

        return (current_x, current_f, iteration,  hist_x, hist_f, lag)

    
    
            

    @profile
    def quasi_newton_dfp(self, init_x, eps=1e-6, store=False):
        self.check_dimension(init_x)
        size = init_x.size

        init_f = self.f(init_x)
        current_x = np.copy(init_x)
        current_g = self.grad_f(current_x)
        current_f = init_f
        current_H = np.eye(size)

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = 0
        lag = 100

        while current_f / init_f > eps:
            current_p = - np.dot(current_H, current_g)

            alpha = line_search(self.f, self.grad_f, current_x, current_p)[0]
            # alpha = self.line_search_wolfe(current_x, current_p, alpha=1, beta=0.9, c1=0.0001, c2=0.9)
            next_x = current_x + alpha * current_p
            next_f = self.f(next_x)
            next_g = self.grad_f(next_x)
            s = alpha * current_p
            y = next_g - current_g

            Hy = current_H.dot(y)

            next_H = current_H \
                - np.outer(Hy, Hy) / np.dot(Hy, y) \
                + np.outer(s, s) / np.dot(s, y)

            if iteration % lag == 0:
                if store:
                    hist_x.append(current_x)
                    hist_f.append(current_f)
                # print "iteration {}: {}".format(iteration, current_f)

            iteration += 1

            current_x = next_x
            current_f = next_f
            current_g = next_g
            current_H = next_H

        return (current_x, current_f, iteration,  hist_x, hist_f, lag)

    def l_bfgs_two_loop(self, grad_f, previous_s, previous_y):
        q = np.copy(grad_f)
        m = len(previous_s)
        previous_alpha = [None] * m

        for i in range(m-1, -1, -1):
            previous_alpha[i] =  np.dot(previous_s[i], q) / np.dot(previous_s[i], previous_y[i])
            q -= previous_alpha[i] * previous_y[i]
        
        gamma = np.dot(previous_s[-1], previous_y[-1]) / np.dot(previous_y[-1], previous_y[-1])
        r =  gamma * np.copy(q)

        for i in range(m):
            beta = np.dot(previous_y[i], r) / np.dot(previous_s[i], previous_y[i])
            r += (previous_alpha[i] - beta) * previous_s[i]
        
        return r

    
    @profile
    def quasi_newton_l_bfgs(self, init_x, eps=1e-6, m=10, store=False):
        self.check_dimension(init_x)
        size = init_x.size

        init_f = self.f(init_x)
        current_x, current_f, current_g, current_H = np.copy(init_x), \
            init_f, self.grad_f(init_x), np.eye(size)

        previous_s = [None] * m
        previous_y = [None] * m

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = 0
        lag = 100

        while current_f / init_f > eps:
            if iteration < m:
                current_p = - np.dot(current_H, current_g)

                alpha = line_search(self.f, self.grad_f, current_x, current_p)[0]
                # alpha = self.line_search_wolfe(current_x, current_p)
                next_x = current_x + alpha * current_p
                next_f = self.f(next_x)
                next_g = self.grad_f(next_x)
                s = alpha * current_p
                y = next_g - current_g

                rho = np.dot(s, y)
                Hy = current_H.dot(y)

                next_H = current_H \
                        + (rho + Hy.dot(y)) * np.outer(s, s) / rho**2 \
                        - (np.outer(Hy, s) + np.outer(s, Hy)) / rho

                previous_s[iteration] = s
                previous_y[iteration] = y

                current_x = next_x
                current_f = next_f
                current_g = next_g
                current_H = next_H
            
            else:
                current_p = - self.l_bfgs_two_loop(current_g, previous_s, previous_y)
                alpha = line_search(self.f, self.grad_f, current_x, current_p)[0]

                next_x = current_x + alpha * current_p
                next_f = self.f(next_x)
                next_g = self.grad_f(next_x)
                s = alpha * current_p
                y = next_g - current_g

                del previous_s[0]
                previous_s.append(s)

                del previous_y[0]
                previous_y.append(y)

                current_x = next_x
                current_f = next_f
                current_g = next_g

            if iteration % lag == 0:
                    if store:
                        hist_x.append(current_x)
                        hist_f.append(current_f)
                    # print "iteration {}: {}".format(iteration, current_f)

            iteration += 1

        return (current_x, current_f, iteration,  hist_x, hist_f, lag)


if __name__ == "__main__":
    # experiment on extended_rosenbrock_function
    x0 = np.array([-1.2, 1] * 800)
    optimizer_erf = Optimizer('erf')

    # # steepest_descent
    # print "---- steepest descent start ----"
    # start = time.time()
    # result_sd = optimizer_erf.steepest_descent(x0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #   result_sd[1], result_sd[2], end - start)
    # print "---- steepest descent end ----\n"


    # # conjugate_gradient_fr
    # print "---- conjugate gradient-fr start ----"
    # start = time.time()
    # result_cg_fr = optimizer_erf.conjugate_gradient_fr(x0, scipy_ls=False)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #     result_cg_fr[1], result_cg_fr[2], end - start)
    # print "---- conjugate gradient-fr end ----\n"

    # conjugate_gradient_pr
    print "---- conjugate gradient-pr start ----"
    start = time.time()
    result_cg_pr = optimizer_erf.conjugate_gradient_pr(x0)
    end = time.time()
    print "minimum: {}\niteration: {}\ntime: {}s".format(
        result_cg_pr[1], result_cg_pr[2], end - start)
    print "---- conjugate gradient-pr end ----\n"

    # # BFGS
    # print "---- quasi_newton_bfgs start ----"
    # start = time.time()
    # result_qn_bfgs = optimizer_erf.quasi_newton_bfgs(x0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #     result_qn_bfgs[1], result_qn_bfgs[2], end - start)
    # print "---- quasi_newton_bfgs end ----\n"

    # L-BFGS
    print "---- quasi_newton_l_bfgs start ----"
    start = time.time()
    result_qn_l_bfgs = optimizer_erf.quasi_newton_l_bfgs(x0)
    end = time.time()
    print "minimum: {}\niteration: {}\ntime: {}s".format(
       result_qn_l_bfgs[1], result_qn_l_bfgs[2], end - start)
    print "---- quasi_newton_L_bfgs end ----\n"

    # # DFP
    # print "---- quasi_newton_dfp start ----"
    # start = time.time()
    # result_qn_dfp = optimizer_erf.quasi_newton_dfp(x0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #     result_qn_dfp[1], result_qn_dfp[2], end - start)
    # print "---- quasi_newton_dfp end ----\n"


    # # experiment on extended_powell_singular_function
    # y0 = np.array([3.0, -1.0, 0.0, 1.0] * 800)
    # optimizer_epsf = Optimizer('epsf')

    # # steepest_descent
    # print "---- steepest descent start ----"
    # start = time.time()
    # result_sd = optimizer_epsf.steepest_descent(y0)
    # end = time.time()
    # print "\nminimum: {}\niteration: {}\ntime: {}s".format(
    #     result_sd[1], result_sd[2], end - start)
    # print "---- steepest descent end ----\n"
    
    # # conjugate_gradient_fr
    # print "---- conjugate gradient-fr start ----"
    # start = time.time()
    # result_cg_fr = optimizer_epsf.conjugate_gradient_fr(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_cg_fr[1], result_cg_fr[2], end - start)
    # print "---- conjugate gradient-fr end ----\n"

    # # conjugate_gradient_pr
    # print "---- conjugate gradient-pr start ----"
    # start = time.time()
    # result_cg_pr = optimizer_epsf.conjugate_gradient_pr(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_cg_pr[1], result_cg_pr[2], end - start)
    # print "---- conjugate gradient-pr end ----\n"


    # # BFGS
    # print "---- quasi_newton_bfgs start ----"
    # start = time.time()
    # result_qn_bfgs = optimizer_epsf.quasi_newton_bfgs(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_qn_bfgs[1], result_qn_bfgs[2], end - start)
    # print "---- quasi_newton_bfgs end ----\n"

    # # L-BFGS
    # print "---- quasi_newton_l_bfgs start ----"
    # start = time.time()
    # result_qn_l_bfgs = optimizer_epsf.quasi_newton_l_bfgs(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_qn_l_bfgs[1], result_qn_l_bfgs[2], end - start)
    # print "---- quasi_newton_L_bfgs end ----\n"

    # # DFP
    # print "---- quasi_newton_dfp start ----"
    # start = time.time()
    # result_qn_dfp = optimizer_epsf.quasi_newton_dfp(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_qn_dfp[1], result_qn_dfp[2], end - start)
    # print "---- quasi_newton_dfp end ----\n"
