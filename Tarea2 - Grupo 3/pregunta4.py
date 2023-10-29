import numpy as np

def gaussian_elimination_complex(M, d):
    """
    The function `gaussian_elimination_complex` performs Gaussian elimination with complex numbers to
    solve a system of linear equations.
    
    :param M: The parameter M is a square matrix representing the coefficients of the system of linear
    equations. Each row of M corresponds to an equation, and each column corresponds to a variable
    :param d: The parameter `d` represents the vector of constants in the system of equations. It is a
    1-dimensional array of complex numbers. Each element of `d` corresponds to the constant term in each
    equation
    :return: the solution vector x, which is a numpy array of complex numbers.
    """
    # Tamaño del sistema de ecuaciones
    n = len(d)
    # Crear la matriz aumentada
    augmented_matrix = np.concatenate((M.astype(np.complex128), d.astype(np.complex128).reshape(-1, 1)), axis=1)

    # Eliminación gaussiana
    for i in range(n):
        # Encuentra la fila con el pivote más grande y realiza intercambio
        pivot_row = max(range(i, n), key=lambda k: abs(augmented_matrix[k, i]))
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Hace ceros debajo del pivote en la columna actual
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Sustitución hacia atrás para encontrar la solución
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j, -1] -= augmented_matrix[j, i] * x[i]

    return x

def qr_factorization(A):
    """
    The `qr_factorization` function performs QR factorization on a given matrix A and returns the
    orthogonal matrix Q and the upper triangular matrix R.
    
    :param A: A is a matrix of shape (m, n)
    :return: two matrices: Q and R.
    """
    # Dimensiones de la matriz A
    m, n = A.shape
    # Matrices Q y R inicializadas
    Q = np.zeros((m, n), dtype=np.complex128)
    R = np.zeros((n, n), dtype=np.complex128)

    # Factorización QR
    for j in range(n):
        v = A[:, j].copy().astype(np.complex128)  # Copia y asegura tipo complejo
        for i in range(j):
            # Calcula las entradas de R y actualiza el vector v
            R[i, j] = np.dot(Q[:, i].conj(), A[:, j])
            v -= R[i, j] * Q[:, i]

        # Calcula la entrada diagonal de R y normaliza el vector v
        R[j, j] = np.linalg.norm(v, 2)
        Q[:, j] = v / R[j, j]

    return Q, R

def question4_example():
    # Definir matrices y vectores de prueba
    W = np.array([[1, 2], [3, 4]])
    T = np.array([[5, 6], [7, 8]])
    p = np.array([9, 10])
    q = np.array([11, 12])

    # Construir la matriz aumentada y el vector de constantes
    A_real = np.block([[W, -T], [T, W]])
    d_real = np.concatenate([p, q])

    # Resolver usando eliminación gaussiana
    sol_gaussian = gaussian_elimination_complex(A_real, d_real)
    print("Solution (Gaussian elimination):", sol_gaussian)

    # Resolver usando factorización QR sin np.linalg.qr
    Q_manual, R_manual = qr_factorization(A_real)
    sol_qr_manual = np.linalg.solve(R_manual, np.dot(Q_manual.T, d_real))
    print("Solution (QR factorization - manual):", sol_qr_manual)

    # Verificar las soluciones convirtiendo a números complejos
    A_complex = W + 1j * T
    x_gaussian = sol_gaussian[:2] + 1j * sol_gaussian[2:]
    x_qr_manual = sol_qr_manual[:2] + 1j * sol_qr_manual[2:]

    # Calcular Ax y comparar con el lado derecho del sistema original
    Ax_gaussian = np.dot(A_complex, x_gaussian)
    Ax_qr_manual = np.dot(A_complex, x_qr_manual)

    # Calcular errores
    b = p + 1j * q
    error_gaussian = np.linalg.norm(Ax_gaussian - b, 2)
    error_qr_manual = np.linalg.norm(Ax_qr_manual - b, 2)

    # Imprimir resultados
    print("Error (Gaussian elimination):", error_gaussian)
    print("Error (QR factorization - manual):", error_qr_manual)

# Ejecutar la función de ejemplo
question4_example()
