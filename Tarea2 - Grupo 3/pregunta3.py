import numpy as np

# Función que implementa el método MHSS
def mhss(A, x0, iter_max, tol):
    """
    The function `mhss` implements the Modified Hermitian and Skew-Hermitian Splitting method for
    solving a linear system of equations.
    
    :param A: The parameter A is a tuple containing two matrices. The first matrix, A[0], represents the
    coefficient matrix in the linear system of equations. The second matrix, A[1], represents the
    constant vector in the linear system of equations
    :param x0: The initial approximation vector
    :param iter_max: The parameter "iter_max" represents the maximum number of iterations that the
    method will perform before stopping
    :param tol: The parameter "tol" stands for tolerance and it represents the desired level of accuracy
    or convergence in the solution. It determines the maximum allowable difference between the actual
    solution and the desired solution. If the difference falls below this tolerance value, the iteration
    process is considered to have converged and the solution is returned
    :return: the final approximation of the solution vector x after performing the specified number of
    iterations.
    """
    # Tamaño del vector x0
    m = len(x0)
    # Matriz identidad del mismo tamaño que x0
    Im = np.eye(m, dtype=complex)
    x = x0

    # Descomposición de la matriz A en W y T
    W, T = np.real(A[0]),  np.imag(A[0])
    # Cálculo de los valores propios de la matriz W
    eigenvalues_W = np.linalg.eigvals(W)
    # Cálculo de alpha* según la fórmula dada
    alpha_star = np.sqrt(np.min(eigenvalues_W) * np.max(eigenvalues_W))
     # Cálculo de M(α) y N(α) según las fórmulas
    M = np.linalg.inv(alpha_star * Im + T).dot((alpha_star * Im + 1j * W).dot(np.linalg.inv(alpha_star * Im + W).dot(alpha_star * Im - 1j * T)))
    N = (1 - 1j) * alpha_star * np.linalg.inv(alpha_star * Im + T).dot(np.linalg.inv(alpha_star * Im + W))
    # Iteración principal del método
    for k in range(iter_max):
        # Cálculo de M(α) y N(α) según las fórmulas
        ##M = np.linalg.inv(alpha_star * Im + T).dot((alpha_star * Im + 1j * W).dot(np.linalg.inv(alpha_star * Im + W).dot(alpha_star * Im - 1j * T)))
        ##N = (1 - 1j) * alpha_star * np.linalg.inv(alpha_star * Im + T).dot(np.linalg.inv(alpha_star * Im + W))

        # Cálculo de la nueva aproximación x(k+1)
        x_new = M.dot(x) + N.dot(A[1])

        # Comprobación del criterio de parada
        if np.linalg.norm(A[0].dot(x_new) - A[1]) < tol:
            return x_new
        x = x_new

    return x

if __name__ == "__main__":
    # Definición de matrices W, T, p, y q
    W = np.array([[12, -2, 6, -2], [-2, 5, 2, 1], [6, 2, 9, -2], [-2, 1, -2, 1]], dtype=np.complex128)
    T = np.array([[6, 2, 7, 2], [2, 7, 1, 1], [7, 1, 9, 0], [2, 1, 0, 10]], dtype=np.complex128)
    p = np.array([[9], [-7], [-5], [7]], dtype=np.complex128)
    q = np.array([[12], [-4], [17], [-2]], dtype=np.complex128)

    # Construcción de la matriz A como una tupla de (W + iT) y (p + iq)
    A = (W + 1j * T, p + 1j * q)
    # Estimación inicial de la solución x0
    x0 = np.array([[1], [-1], [1j], [-1j]], dtype=complex)
    iter_max = 1000  # Número máximo de iteraciones
    tol = 1e-12  # Tolerancia para el criterio de parada

    # Ejecución del método MHSS para encontrar la solución
    solution = mhss(A, x0, iter_max, tol)

    # Impresión de la solución encontrada
    print("Solución encontrada:")
    print(solution)
    print("Solución real:")
    real_solution = np.array([[1], [-1], [1j], [-1j]], dtype=complex)
    print(real_solution)
    # Cálculo y muestra del valor óptimo de alpha*
    eigenvalues_W = np.linalg.eigvals(W)
    alpha_star = np.sqrt(np.min(eigenvalues_W) * np.max(eigenvalues_W))
    print(alpha_star)

    
    