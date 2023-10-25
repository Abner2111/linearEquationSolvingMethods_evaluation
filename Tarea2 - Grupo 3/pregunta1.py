import numpy as np
import time

def hss_iteration(W, T, b, x0, max_iter, tol):
    # Inicializar la matriz identidad
    Im = np.eye(len(W))
    # Inicializar el vector de solución actual
    x = x0

    # Iniciar el contador de tiempo
    start_time = time.time()

    # Iterar hasta alcanzar el número máximo de iteraciones
    for k in range(max_iter):
        # Calcular z según la fórmula del método HSS
        z = np.linalg.solve(Im + W, np.linalg.solve(Im - T, x) + np.linalg.solve(Im + W, b))
        # Calcular la nueva solución x_new según la fórmula del método HSS
        x_new = np.linalg.solve(Im + T, z)

        # Calcular el error actual
        error = np.linalg.norm(np.dot(W + T, x_new) - b, 2)

        # Verificar si el error es menor que la tolerancia
        if error < tol:
            # Registrar el tiempo de ejecución
            end_time = time.time()
            execution_time = end_time - start_time
            # Devolver la solución, el error, el tiempo de ejecución y el número de iteraciones
            return x_new, error, execution_time, k + 1

        # Actualizar la solución actual
        x = x_new

    # Si no converge, registrar el tiempo de ejecución y devolver los resultados con el número máximo de iteraciones
    end_time = time.time()
    execution_time = end_time - start_time
    return x, error, execution_time, max_iter

# Definir matrices y vectores del sistema
W = np.array([[12, -2, 6, -2],
              [-2, 5, 2, 1],
              [6, 2, 9, -2],
              [-2, 1, -2, 1]])

T = np.array([[6, 2, 7, 2],
              [2, 7, 1, 1],
              [7, 1, 9, 0],
              [2, 1, 0, 10]])

b = np.array([9, -7, -5, 7])
x0 = np.array([0, 0, 0, 0])

# Aplicar el método HSS
solution, error, execution_time, iterations = hss_iteration(W, T, b, x0,1000 ,1e-12)

# Imprimir los resultados
print("Método 1: HSS")
print("error =", "{:.8e}".format(error))
print("Tiempo de ejecución =", "{:.8f}".format(execution_time), "segs")
print("Iteraciones =", iterations)
print("Solución aproximada usando HSS:", solution)
print("Solución exacta:", [1, -1, 1j, -1j])
