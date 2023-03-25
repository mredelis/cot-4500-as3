import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

###### Exercise 1: Euler's Method ######
def function(t: float, w: float):
    # function is y' = t - y**2 so f(ti, wi) = ti - wi**2
    return t - (w**2)


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


###### Exercise 2: Runge-Kutta (Midpoint method) with same function as Euler's ######
def midPoint() -> float:

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
        w = prev_w + h * function(prev_t + h / 2, prev_w + h / 2 * function(prev_t, prev_w))
        # compute ti
        t = a + i * h
        # print(w)

    return w


###### Exercise 3: ######
# Use Gaussian elimination and backward substitution solve a linear system of equations written in augmented matrix format.

###### Exercise 5: Find if a matrix is diagonally dominant ######
def diagonally_dominant_matrix(A: np.array) -> float:
    # check if matrix A is square n x n
    (row, column) = np.shape(A)
    if row != column:
        print("Matrix must be of type nxn (square matrix)")
        return False

    # traverse the rows
    for i in range(row):
        sum = 0
        # traverse columns
        for j in range(column):
            if i == j:
                diagonal_ele = abs(A[i, j])
            else:
                sum += abs(A[i, j])

        if diagonal_ele < sum:
            return False

    return True


###### Exercise 6: Find if a matrix is diagonally dominant ######
def positive_definite_matrix(A: np.array) -> float:
    return True


#########################################################
print()

approximate_euler_solution = eulers()
print(f"Euler's Method Approximate Solution: {approximate_euler_solution}\n")

approximate_midpoint_solution = midPoint()
print(f"Runge-Kutta Method Approximate Solution: {approximate_midpoint_solution}\n")

A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
print(A)
# b here is a row vector but it should be a column vector
b = np.array([6, 0, 3])
print(b)


# Exercise 5 Diagonally Dominant Matrix
# Examples from textbook
# matrix = np.array([[7, 2, 0], [3, 5, -1], [0, 5, -6]])
# matrix = np.array([[6, 4, -3], [4, -2, 0], [-3, 0, 1]])
matrix = np.array(
    [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8],
    ]
)
print(f"Diagonally Dominant Matrix?: {diagonally_dominant_matrix(matrix)}")

# Exercise 6 Positive Definite Matrix
mat = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])

print(f"Positive Definite Matrix?: {positive_definite_matrix(mat)}")

# print(mat)
