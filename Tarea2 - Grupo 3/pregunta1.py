import numpy as np
import time

def hss(A, b, x0, exact_solution, max_iter, tol):
    """
    The function `hss` implements the HSS (Hilbert-Schmidt Smoothing) method to solve a linear system of
    equations.
    
    :param A: A is a square matrix of size n x n. It represents the coefficient matrix of a linear
    system of equations
    :param b: The parameter "b" represents the right-hand side vector in the linear system of equations
    Ax = b. It is a column vector of size n, where n is the number of unknowns in the system
    :param x0: The initial guess for the solution vector x
    :param exact_solution: The exact_solution parameter is the known solution to the linear system of
    equations Ax = b. It is used to calculate the error in the solution obtained by the HSS method
    :param max_iter: The maximum number of iterations allowed for the HSS method
    :param tol: tol is the tolerance level for convergence. It is the maximum acceptable difference
    between the calculated solution and the exact solution. If the difference is below this tolerance
    level, the method is considered to have converged
    :return: the following variables: x (the approximate solution), k + 1 (the number of iterations
    performed), elapsed_time (the time taken to execute the method), error (the error in comparison with
    the exact solution), and exact_solution (the provided exact solution).
    """
    # Inicialización de variables
    n = A.shape[0]
    x = x0

    # Inicialización de variables para medir el tiempo
    start_time = time.time()

    # Iteraciones del método HSS
    for k in range(max_iter):
        # Cálculo del residuo
        r = b - np.dot(A, x)

        # Cálculo de la corrección z mediante la resolución de Ax = r
        z = np.linalg.solve(A, r)

        # Actualización de x con la corrección z
        x = x + z

        # Verificación de convergencia utilizando la norma Euclidiana
        if np.linalg.norm(np.dot(A, x) - b, 2) <= tol:
            # Cálculo del tiempo de ejecución
            elapsed_time = time.time() - start_time

            # Cálculo del error en comparación con la solución exacta
            error = np.linalg.norm(x - exact_solution, 2)

            # Devolver variables por separado
            return x, k + 1, elapsed_time, error, exact_solution

    # Si no converge en max_iter iteraciones
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(x - exact_solution, 2)
    return x, max_iter, elapsed_time, error, exact_solution

# Definición de matrices y parámetros
W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1], [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=complex)
T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0], [2, 1, 0, 10]], dtype=complex)
p = np.array([9, -7, -5, 7], dtype=complex)
q = np.array([12, -4, 17, -2], dtype=complex)

# Construcción de la matriz A y el vector b
A = W + 1j * T
b = p + 1j * q

# Estimación inicial y parámetros del método
x0 = np.array([0, 0, 0, 0], dtype=complex)
iter_max = 1000
tol = 1e-12

# Solución exacta proporcionada
exact_solution = np.array([1, -1, 1j, -1j], dtype=complex)

# Llamada al método HSS
solution, iterations, elapsed_time, error, _ = hss(A, b, x0, exact_solution, iter_max, tol)

# Impresión de resultados
print("Solución aproximada:", solution)
print("Iteraciones realizadas:", iterations)
print("Tiempo de ejecución:", elapsed_time)
print("Error en comparación con la solución exacta:", error)
print("Solución exacta:", exact_solution)
