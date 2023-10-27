import numpy as np

#                                              _________________________________________________________
# ___________________________________________/Efficient parameterized HSS iteration methods- algorithm 2


def PNHSS(W, T, p, q, alpha, omega, max_iter, tol, x0):
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
