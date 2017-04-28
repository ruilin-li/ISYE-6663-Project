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
    def quasi_newton_bfgs(self, init_x, eps=1e-6, store=False):
        self.check_dimension(init_x)
        size = init_x.size

        init_f = self.f(init_x)
        current_x, current_f, current_g, current_H = np.copy(init_x), \
            init_f, self.grad_f(init_x), np.eye(size)

        hist_x = [init_x]
        hist_f = [init_f]
        m = 10
        previous_s = [None] * m
        previous_y = [None] * m

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
            
            if iteration < m:
                previous_s[iteration] = s
                previous_y[iteration] = y

            iteration += 1

            current_x = next_x
            current_f = next_f
            current_g = next_g
            current_H = next_H

        return (current_x, current_f, iteration, previous_s, previous_y)


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
    def quasi_newton_l_bfgs(self, init_x, previous_s, previous_y, eps=1e-6, m=10, store=False):
        self.check_dimension(init_x)
        size = init_x.size

        init_f = self.f(init_x)
        current_x, current_f, current_g = np.copy(init_x), init_f, self.grad_f(init_x)

        hist_x = [init_x]
        hist_f = [init_f]

        iteration = m
        lag = 100

        while current_f / init_f > eps:
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
    x0 = np.array([-1.2, 1] * 1600)
    optimizer_erf = Optimizer('erf')

   
    # BFGS
    print "---- quasi_newton_bfgs start ----"
    start = time.time()
    result_qn_bfgs = optimizer_erf.quasi_newton_bfgs(x0)
    end = time.time()
    print "minimum: {}\niteration: {}\ntime: {}s".format(
        result_qn_bfgs[1], result_qn_bfgs[2], end - start)
    print "---- quasi_newton_bfgs end ----\n"

    previous_s = result_qn_bfgs[3]
    previous_y = result_qn_bfgs[4]

    # L-BFGS
    print "---- quasi_newton_l_bfgs start ----"
    start = time.time()
    result_qn_l_bfgs = optimizer_erf.quasi_newton_l_bfgs(x0, previous_s, previous_y)
    end = time.time()
    print "minimum: {}\niteration: {}\ntime: {}s".format(
       result_qn_l_bfgs[1], result_qn_l_bfgs[2], end - start)
    print "---- quasi_newton_L_bfgs end ----\n"


    # experiment on extended_powell_singular_function
    y0 = np.array([3.0, -1.0, 0.0, 1.0] * 800)
    optimizer_epsf = Optimizer('epsf')

    
    # # BFGS
    # print "---- quasi_newton_bfgs start ----"
    # start = time.time()
    # result_qn_bfgs = optimizer_epsf.quasi_newton_bfgs(y0)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_qn_bfgs[1], result_qn_bfgs[2], end - start)
    # print "---- quasi_newton_bfgs end ----\n"

    # previous_s = result_qn_bfgs[3]
    # previous_y = result_qn_bfgs[4]
    
    # # L-BFGS
    # print "---- quasi_newton_l_bfgs start ----"
    # start = time.time()
    # result_qn_l_bfgs = optimizer_epsf.quasi_newton_l_bfgs(y0, previous_s, previous_y)
    # end = time.time()
    # print "minimum: {}\niteration: {}\ntime: {}s".format(
    #    result_qn_l_bfgs[1], result_qn_l_bfgs[2], end - start)
    # print "---- quasi_newton_L_bfgs end ----\n"
