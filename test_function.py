import numpy as np
from function import extended_rosenbrock_function as f1
from project_messy import extended_rosenbrock_function as f2
from function import extender_powell_singular_function as g1
from project_messy import extender_powell_singular_function as g2
from function import gradient_erf as grad_f1
from function import gradient_epsf as grad_g1
from project_messy import gradient

eps = 1e-6
number = 100
test_points = np.random.normal(scale=1, size=(number, 4))

# test extended_rosenbrock_function
error = 0
for point in test_points:
    if np.linalg.norm(f1(point) - f2(point)) > eps:
        error += 1

if error == 0:
    print "---- function 1 pass test ----"
else:
    print error

# test extender_powell_singular_function
error = 0
for point in test_points:
    if np.linalg.norm(g1(point) - g2(point)) > eps:
        error += 1

if error == 0:
    print "---- function 2 pass test ----"
else:
    print error

# test gradient of extended_rosenbrock_function
error = 0
for point in test_points:
    if np.linalg.norm(gradient(f1, point) - grad_f1(point)) > eps:
        error += 1

if error == 0:
    print "---- function 1 gradient pass test ----"
else:
    print error

# test gradient of extender_powell_singular_function
error = 0
for point in test_points:
    if np.linalg.norm(gradient(g1, point) - grad_g1(point)) > eps:
        error += 1
        print gradient(g1, point)
        print grad_g1(point)

if error == 0:
    print "---- function 2 gradient pass test ----"
else:
    print error
