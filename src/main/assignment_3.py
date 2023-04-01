import numpy as np
from numpy.linalg import eig

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


###### Exercise 2: Runge-Kutta (method of order four) ######
def runge_kutta() -> float:

    # Initial setup
    w = 1  # w = α for this exercise the initial condition is y(a) = α
    a, b = (0, 2)  # t is in the interval [a, b]
    t = a
    N = 10  # number of iterations
    h = (b - a) / N  # step size

    for i in range(1, N + 1):
        prev_w = w
        prev_t = t

        # compute wi. Breakdown the solution below
        K1 = h * function(prev_t, prev_w)
        K2 = h * function((prev_t + h / 2), (prev_w + K1 / 2))
        K3 = h * function((prev_t + h / 2), (prev_w + K2 / 2))
        K4 = h * function((prev_t + h), (prev_w + K3))
        w = prev_w + (K1 + 2 * K2 + 2 * K3 + K4) / 6

        # compute ti
        t = a + i * h

    return w


###### Exercise 3: Gaussian elimination and backward substitution ######
# Solve a linear system of equations written in augmented matrix format.
def gaussian_elimination(A: np.array, b: np.array):
    n = len(b)

    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # numpy array of n size and initializing to zero for storing solution vector
    x = np.zeros(n)

    # Perform elimination
    for i in range(n):
        # Find pivot row to move the entry with largest abs value to the pivot position
        max_row = i

        for j in range(i + 1, n):
            if abs(Ab[j, i]) > abs(Ab[max_row, i]):
                max_row = j

        # Swap rows to bring pivot element to diagonal
        # Selects a submatrix consisting of all rows i to max_row
        Ab[[i, max_row], :] = Ab[[max_row, i], :]

        # Eliminate entries below pivot
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, :] = Ab[j, :] - factor * Ab[i, :]

        if Ab[n - 1, n - 1] == 0:
            print("No unique solution exists")
            return

    # Start backward substitution
    x[n - 1] = Ab[n - 1, n] / Ab[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - Ab[i][j] * x[j]

        x[i] = x[i] / Ab[i][i]

    return x


###### Exercise 4: LU Factorization ######
def LU_factorization(mat: np.array):
    # matrix determinant. It has to be a square matrix. Non-square matrix does not have det
    print(f"{np.linalg.det(mat)}\n")

    # print(np.shape(mat)) # gives (rows, columns)
    # print(len(mat)) # gives the number of rows same as mat.shape[0]
    n = len(mat)

    # Initialize U to an identity matrix of dimension n x n
    U = np.identity(n)
    # Initialize L = A
    L = mat.copy()

    # Find L and U matrices
    for i in range(n):
        for j in range(i + 1, n):
            if L[i, i] != 0:  # check for div by 0 error
                factor = L[j, i] / L[i, i]
                U[j, i] = factor
                L[j, :] = L[j, :] - factor * L[i, :]
            else:
                print("Division by 0 error!")
                return

    print(f"{U}\n\n{L}\n")


###### Exercise 5: Find if a matrix is diagonally dominant ######
def diagonally_dominant_matrix(A: np.array):
    # check if matrix A is square n x n
    (row, column) = np.shape(A)
    if row != column:
        print("Matrix must be of type nxn (square matrix)")
        return False

    # traverse the rows
    for i in range(row):
        sum = 0
        # traverse columns and find the sum of each row
        for j in range(column):
            sum += abs(A[i, j])

        # removing diagonal element
        sum = sum - abs(A[i, i])

        # checking if diagonal element is less than sum of non-diagonal element.
        if abs(A[i, i]) < sum:
            return False

    return True


###### Exercise 6: Find if a matrix is positive definite ######
# if it is symmetric and all its eigenvalues λ are positive, that is λ > 0


def is_Symmetric(A: np.array) -> bool:
    # Transpose the matrix
    B = A.transpose()
    # check if both the arrays are of equal size
    if A.shape == B.shape:
        # comparing the arrays using == and all() method
        if (A == B).all():
            return True
        else:
            return False
    else:
        return False


def positive_eigenvalues(A: np.array) -> bool:
    # w are eigen values and v are eigen vectors
    w, v = eig(A)
    return all(i > 0 for i in w)


def positive_definite_matrix(A: np.array):
    return is_Symmetric(A) and positive_eigenvalues(A)


##############################################################################
# Exercise 1 Euler method
approximate_euler_solution = eulers()
print("%.5f" % approximate_euler_solution)
print()

# Exercise 2 Runge-Kutta method
approximate_rungeKutta_solution = runge_kutta()
print("%.5f" % approximate_rungeKutta_solution)
print()

# Exercise 3 Gaussian elimination with backward substitution
A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
b = np.array([6, 0, -3])  # b is a row vector but it should be a column vector
x = gaussian_elimination(A, b)
print(f"{x}\n")

# Exercise 4 LU Factorization
mat = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
LU_factorization(mat)

# Exercise 5 Diagonally Dominant Matrix
matrix = np.array(
    [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8],
    ]
)
print(diagonally_dominant_matrix(matrix))
print()

# Exercise 6 Positive Definite Matrix
mat = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
print(positive_definite_matrix(mat))
