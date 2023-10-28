import numpy as np

def mhss(A, b, x0, max_iter=1000, tol=1e-12):
    # Inicialización de variables
    n = A.shape[0]
    x = x0

    # Iteraciones del método MHSS
    for k in range(max_iter):
        r = b - np.dot(A, x)  # Residuo
        z = np.linalg.solve(A + np.conj(A.T), r)  # Cálculo de z
        x = x + z  # Actualización de x
        
        if np.linalg.norm(np.dot(A, x) - b) < tol:  # Verificación de convergencia
            return x
    
    return x

def calculate_optimal_alpha(W):
    eigenvalues_W = np.linalg.eigvals(W)
    alpha_star = np.sqrt(np.min(eigenvalues_W) * np.max(eigenvalues_W))
    return alpha_star

# Definición de matrices y parámetros
W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1], [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=complex)
T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0], [2, 1, 0, 10]], dtype=complex)
p = np.array([9, -7, -5, 7], dtype=complex)
q = np.array([12, -4, 17, -2], dtype=complex)

# Construcción de la matriz A y el vector b
A = W + 1j * T
b = p + 1j * q

# Estimación inicial y parámetros del método
x0 = np.array([1, -1, 1j, -1j], dtype=complex)
iter_max = 1000
tol = 1e-12

# Ejecución del método MHSS
solution = mhss(A, b, x0, iter_max, tol)

# Impresión de la solución encontrada y la solución real
print("Solución encontrada:")
print(solution)
print("Solución real:")
real_solution = np.array([1, -1, 1j, -1j], dtype=complex)
print(real_solution)

# Cálculo y muestra del valor óptimo de alpha*
alpha_star = calculate_optimal_alpha(W)
print("Valor óptimo de alpha*:", alpha_star)
