import numpy as np

def mhss(A, b, x0, max_iter=1000, tol=1e-12):
    n = A.shape[0]
    x = x0

    for k in range(max_iter):
        r = b - np.dot(A, x)
        z = np.linalg.solve(A + np.conj(A.T), r)
        x = x + z
        
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x
    
    return x

def calculate_optimal_alpha(W):
    eigenvalues_W = np.linalg.eigvals(W)
    alpha_star = np.sqrt(np.min(eigenvalues_W) * np.max(eigenvalues_W))
    return alpha_star

W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1], [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=complex)
T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0], [2, 1, 0, 10]], dtype=complex)
p = np.array([9, -7, -5, 7], dtype=complex)
q = np.array([12, -4, 17, -2], dtype=complex)

A = W + 1j * T
b = p + 1j * q
x0 = np.array([1, -1, 1j, -1j], dtype=complex)
iter_max = 1000
tol = 1e-12
solution = mhss(A, b, x0, iter_max, tol)

print("Solución encontrada:")
print(solution)
print("Solución real:")
real_solution = np.array([1, -1, 1j, -1j], dtype=complex)
print(real_solution)

alpha_star = calculate_optimal_alpha(W)
print("Valor óptimo de alpha*:", alpha_star)
