import numpy as np

#                                              _________________________________________________________
# ___________________________________________/Efficient parameterized HSS iteration methods- algorithm 2


def PNHSS(W, T, p, q, alpha, omega, max_iter, tol, x0):
    """
    The function PNHSS implements the Preconditioned Non-Hermitian Subspace Shift method for solving a
    linear system of equations.
    
    :param W: A matrix of shape (n, n) representing the weight matrix in the PNHSS algorithm
    :param T: The parameter T represents a square matrix in the PNHSS function
    :param p: The parameter `p` is a numpy array representing the real part of a complex vector
    :param q: The parameter "q" represents a vector in the PNHSS function. It is used in the calculation
    of the variable "B" in the first step and the variable "D" in the second step
    :param alpha: Alpha is a scalar parameter used in the calculation of matrices A and C. It is
    multiplied by the identity matrix I in both cases
    :param omega: The parameter omega is a scalar value that determines the weight of the W matrix in
    the calculations. It is used to balance the influence of the W matrix and the T matrix in the
    algorithm
    :param max_iter: The maximum number of iterations allowed for the algorithm to converge. If the
    algorithm does not converge within this number of iterations, it will stop and return the current
    solution
    :param tol: tol is the tolerance level for convergence. It is the maximum acceptable error between
    the current solution and the previous solution. If the error falls below this tolerance level, the
    algorithm is considered to have converged and the iteration stops
    :param x0: The initial guess for the solution vector x
    :return: three values: x_k, which is the solution vector, error, which is the error between the
    current solution and the previous solution, and max_iter, which is the maximum number of iterations
    performed.
    """
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = np.zeros(n, dtype=complex)

    for k in range(max_iter):
        # Primer paso
        A = omega * W + T
        B = -1j * (omega * T - W) @ x_k + (omega - 1j) * (p + 1j * q)
        x_half = np.linalg.solve(A, B)

        # Segundo paso
        C = alpha * I + omega * W + T
        D = (alpha * I - 1j * (omega * T - W)) @ x_half + \
            (omega - 1j) * (p + 1j * q)
        x_k = np.linalg.solve(C, D)

        # Calcular el error
        error = np.linalg.norm(x_k - x_half)

        if error < tol:
            return x_k, error, max_iter

    return x_k, error, max_iter

#                                              _________________________________________________________
# ___________________________________________/Efficient parameterized HSS iteration methods- algorithm 3


def PSHSS(W, T, p, q, alpha, omega, max_iter, tol, x0):
    """
    The function PSHSS solves a linear system of equations using the Preconditioned Shifted Hermitian
    and Skew-Hermitian Splitting method.
    
    :param W: W is a square matrix of size n x n
    :param T: T is a square matrix of size n x n
    :param p: The parameter `p` is a numpy array representing the real part of a complex vector
    :param q: The parameter "q" is not explicitly defined in the code snippet you provided. It is used
    as an input to the function `PSHSS`, but its purpose and expected format are not clear. Could you
    please provide more information about the parameter "q" and its intended use in the code?
    :param alpha: Alpha is a scalar value used in the calculation of the matrix X. It is multiplied by
    the identity matrix I before being added to the other matrices W and T
    :param omega: The parameter "omega" is a scalar value used in the PSHSS function. It is used in the
    calculation of the matrix X, which is the sum of three matrices: alpha times the identity matrix,
    omega times matrix W, and matrix T. The value of omega determines the relative importance of
    :param max_iter: The maximum number of iterations allowed for the PSHSS algorithm
    :param tol: tol is the tolerance level for the error. It is the maximum acceptable difference
    between the current solution and the previous solution. If the error falls below this tolerance
    level, the iteration process will stop and the current solution will be returned
    :param x0: The initial guess for the solution vector x
    :return: three values: x_k, which is the solution vector, error, which is the error between the
    current solution and the previous solution, and max_iter, which is the maximum number of iterations
    performed.
    """
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = np.zeros(n, dtype=complex)

    for k in range(max_iter):
        X = alpha * I + omega * W + T
        Z = (alpha * I - 1j * (omega * T - W)) @ x_k + \
            (omega - 1j) * (p + 1j * q)
        x_k = np.linalg.solve(X, Z)

        # Calcular el error
        error = np.linalg.norm(x_k - x_half)

        if error < tol:
            return x_k, error, max_iter

    return x_k, error, max_iter


# Parámetros
W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1],
             [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=complex)
T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0],
             [2, 1, 0, 10]], dtype=complex)
p = np.array([9, -7, -5, 7], dtype=complex)
q = np.array([12, -4, 17, -2], dtype=complex)
alpha = 1
omega = 1
max_iter = 1000
tol = 1e-12
x0 = np.zeros(len(p), dtype=complex)

# Soluciones utilizando los diferentes métodos
solution1, error1, iter1 = PNHSS(W, T, p, q, alpha, omega, max_iter, tol, x0)
print("Solución utilizando el método PNHSS:")
print("Solución (PNHSS):", solution1)
print("Error (PNHSS):", error1)
print("Número de iteraciones (PNHSS):", iter1)

solution2, error2, iter2 = PSHSS(W, T, p, q, alpha, omega, max_iter, tol, x0)
print("Solución utilizando el método PSHSS:")
print("Solución (PSHSS):", solution2)
print("Error (PSHSS):", error2)
print("Número de iteraciones (PSHSS):", iter2)
