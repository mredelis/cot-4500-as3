import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

###### Exercise 1: Euler's Method ######


def eulers() -> float:

    # Initial setup
    w = 1  # w0 = y(0) for this exercise the initial condition is f(0) = 1 = w0
    a, b = (0, 2)  # t is in the interval [a, b]
    t = a
    N = 10  # number of iterations
    h = (b - a) / N  # step size

    for i in range(1, N + 1):
        prev_w = w
        prev_t = t
        # compute wi
        w = prev_w + h * function(prev_t, prev_w)
        # compute ti
        t = a + i * h
        # print(w)

    return w


def function(t: float, w: float):
    # function is y' = t - y**2 so f(ti, wi) = ti - wi**2
    return t - (w**2)


#########################################################
print()

approximate_solution = eulers()
print(f"Euler's Method Approximate Solution: {approximate_solution}\n")
