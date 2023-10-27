import numpy as np

def gaussian_elimination(M, d):
    # Realiza la eliminación gaussiana
    n = len(d)
    augmented_matrix = np.concatenate((M.astype(np.float64), d.astype(np.float64).reshape(-1, 1)), axis=1)

    for i in range(n):
        # Pivotea la matriz
        pivot_row = max(range(i, n), key=lambda k: abs(augmented_matrix[k, i]))
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Elimina por debajo del pivote
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j, -1] -= augmented_matrix[j, i] * x[i]

    return x

def qr_factorization(M):
    # Realiza la factorización QR
    Q, R = np.linalg.qr(M.astype(np.float64))
    return Q, R

# Ejemplo de uso
W = np.array([[1, 2], [3, 4]])
T = np.array([[5, 6], [7, 8]])
p = np.array([9, 10])
q = np.array([11, 12])

A = W + 1j * T
b = p + 1j * q

# Crea el sistema de valores reales
M = np.block([[W, -T], [T, W]])
d = np.concatenate([p, q])

# Resuelve usando eliminación gaussiana
solucion_gaussiana = gaussian_elimination(M, d)
print("Solución (eliminación gaussiana):", solucion_gaussiana)

# Resuelve usando factorización QR
Q, R = qr_factorization(M)
solucion_qr = np.linalg.solve(R, np.dot(Q.T, d))
print("Solución (factorización QR):", solucion_qr)

# Verifica la solución
Ax = np.dot(A, solucion_gaussiana[:2])  # Ajusta los índices según el tamaño de las matrices
error = np.linalg.norm(Ax - b, 2)
print("Error:", error)
