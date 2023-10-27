import numpy as np

def MHSS(A, x0, iter_max, tol):
    m = len(x0)
    Im = np.eye(m, dtype=complex)
    x = x0

    W, T = A
    alpha_star = np.sqrt(np.min(np.linalg.eigvals(W)) * np.max(np.linalg.eigvals(W)))


    for k in range(iter_max):
        M = np.linalg.inv(alpha_star * Im + T).dot((alpha_star * Im + 1j * W).dot(np.linalg.inv(alpha_star * Im + W).dot(alpha_star * Im - 1j * T)))
        N = (1 - 1j) * alpha_star * np.linalg.inv(alpha_star * Im + T).dot(np.linalg.inv(alpha_star * Im + W))

        x_new = M.dot(x) + N.dot(A[1])
        if np.linalg.norm(A[0].dot(x_new) - x0) < tol:
            return x_new
        x = x_new

    return x

if __name__ == "__main__":
    W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1], [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=complex)
    T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0], [2, 1, 0, 10]], dtype=complex)
    p = np.array([[9], [-7], [-5], [7]], dtype=complex)
    q = np.array([[12], [-4], [17], [-2]], dtype=complex)
    A = (W + 1j * T, p + 1j * q)
    x0 = np.array([[1], [-1], [0], [-0]], dtype=complex)
    iter_max = 1000
    tol = 1e-12

    solution = MHSS(A, x0, iter_max, tol)

    print("Solución encontrada:")
    print(solution)
    
