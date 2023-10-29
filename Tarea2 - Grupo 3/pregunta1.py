import numpy as np
import time

def hss(A, b, x0, max_iter=1000, tol=1e-12):
    # Inicialización de variables
    n = A.shape[0]
    x = x0

    # Inicialización de variables para medir el tiempo
    start_time = time.time()

    # Iteraciones del método HSS
    for k in range(max_iter):
        r = b - np.dot(A, x)  # Residuo
        z = np.linalg.solve(A, r)  # Cálculo de z
        x = x + z  # Actualización de x

        # Verificación de convergencia
        if np.linalg.norm(np.dot(A, x) - b, 2) <= tol:
            # Cálculo del tiempo de ejecución
            elapsed_time = time.time() - start_time

            # Cálculo del error
            error = np.linalg.norm(np.dot(A, x) - b, 2)

            # Devolver variables por separado
            return x, k + 1, elapsed_time, error, np.linalg.solve(A, b)

    # Si no converge en max_iter iteraciones
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(np.dot(A, x) - b, 2)
    return x, max_iter, elapsed_time, error, np.linalg.solve(A, b)


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



solution, iterations, elapsed_time, error, exact_solution = hss(A, b, x0, iter_max, tol)
print("Solución aproximada:", solution)
print("Iteraciones realizadas:", iterations)
print("Tiempo de ejecución:", elapsed_time)
print("Error:", error)
print("Solución exacta:", exact_solution)
