import time
import numpy as np
ITERMAX = 5000
TOL = 1e-6
"""
Matriz tridiagonal de mxm , con diagonal = b, 
elemento a la izquierda de la diagonal = a
y elemento a la derecha de la diagonal = c


ejm: tridiagMatrix(-1,2,-2,6)
[[ 2. -1.  0.  0.  0.  0.]
 [-1.  2. -1.  0.  0.  0.]
 [ 0. -1.  2. -1.  0.  0.]
 [ 0.  0. -1.  2. -1.  0.]
 [ 0.  0.  0. -1.  2. -1.]
 [ 0.  0.  0.  0. -1.  2.]]
"""
def tridiagMatrix(a,b,c,m):
    A = np.zeros((m,m))
    n = len(A)

    A[0][0] = b
    A[0][1] = c

    A[n-1][n-1]=b
    A[n-1][n-2]=a

    for i in range (1,n-1):
        A[i][i-1] = a
        A[i][i] = b
        A[i][i+1] = c
    return A

def k(m,h,Im):
   
    Bm = (1/(h*h))*tridiagMatrix(-1,2,-1,m)
    
    K = np.kron(Im, Bm) + np.kron(Bm,Im)
    return K
"""
Sistema lineal complejo  (W+iT)x = b
Argumentos: m
Retorna: W y T como matrices de m^2*m^2, 
         b como vector complejo de m^2
"""
def W_T_b(m):
    h = 1/(m+1)
    Im = np.identity(m)
    K = k(m, h, Im)
    W = K +((3-np.sqrt(3))/h)
    T = K + ((3+np.sqrt(3))/h)
    b = np.zeros((m*m,1),dtype=np.complex_)
    for i in range(0,m*m):
        b[i][0] = ((1-1j)*i)/((1j+1)**2)

    return W*(h*h),T*(h*h),b*(h*h)



#-----------------------------------------------------------------------
#METODOS DE SOLUCION DE SISTEMAS LINEALES

def hss(A, b, x0,  max_iter=1000, tol=1e-12):
    """
    The function `hss` implements the HSS (Hermitian and Skew-Symmetric) method for solving a linear
    system of equations.
    
    :param A: A is a square matrix of size n x n
    :param b: The parameter `b` represents the right-hand side vector in the linear system of equations
    `Ax = b`, where `A` is the coefficient matrix and `x` is the unknown vector. It contains the values
    on the right-hand side of each equation
    :param x0: The initial guess for the solution vector x
    :param max_iter: The maximum number of iterations allowed for the HSS method. If the method does not
    converge within this number of iterations, it will stop and return the current solution, defaults to
    1000 (optional)
    :param tol: The parameter "tol" stands for tolerance and it is used to determine the convergence
    criteria for the method. It represents the maximum acceptable error between the computed solution
    and the actual solution. If the error falls below this tolerance, the method is considered to have
    converged
    :return: four variables: x, iteraciones, elapsed_time, and error.
    """
    # Inicialización de variables
    n = A.shape[0]
    x = x0
    #identiti
    Im = np.identity(n)

    
    # inverse of (Im+iT)
    Ki = np.linalg.solve(Im+1j*np.imag(A),Im)

    #Im-W
    H = Im-np.real(A)

    #inverse of Im+W
    Oi = np.linalg.solve(Im+np.real(A),Im)

    #Im-iT
    L = Im-1j*np.imag(A)
    # Inicialización de variables para medir el tiempo
    start_time = time.time()
    iteraciones = 0
    # Iteraciones del método HSS
    for k in range(max_iter):
        # Cálculo del residuo
        z_k = np.dot(np.dot(Oi,L),x) + np.dot(Oi,b)

        x = np.dot(np.dot(Ki,H),z_k) + np.dot(Ki,b)

        iteraciones = k+1
        # Verificación de convergencia utilizando la norma Euclidiana
        if np.linalg.norm(np.dot(A, x) - b) <= np.linalg.norm(b)*tol:
            # Cálculo del tiempo de ejecución
            elapsed_time = time.time() - start_time

            # Cálculo del error en comparación con la solución exacta
            error = np.linalg.norm(np.dot(A, x) - b)
            

            # Devolver variables por separado
            return x, iteraciones, elapsed_time, error

    # Si no converge en max_iter iteraciones
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(np.dot(A, x) - b)
    return x, max_iter, elapsed_time, error


def PNHSS(W, T, p, q,max_iter, tol, x0):
    """
    The function `PNHSS` implements the Preconditioned Non-Hermitian Symmetric Successive Overrelaxation
    (PNHSS) method to solve a linear system of equations.
    
    :param W: W is a square matrix of size n x n, representing the real part of the coefficient matrix
    in a linear system of equations
    :param T: The parameter T represents a matrix in the PNHSS function. It is used in the calculations
    of the first and second steps of the algorithm
    :param p: The parameter `p` represents the real part of the right-hand side vector in a linear
    system of equations. It is a numpy array of shape (n,), where n is the size of the system
    :param q: The parameter `q` represents the imaginary part of the vector `b` in the equation `Ax =
    b`, where `A` is a matrix and `x` is the unknown vector
    :param max_iter: The maximum number of iterations allowed for the algorithm to converge
    :param tol: tol is the tolerance level for convergence. It is used to determine when to stop the
    iteration process. If the norm of the difference between the product of matrix Am and vector x_k and
    vector b is less than tol times the norm of vector b, the iteration process is considered converged
    and stopped
    :param x0: The initial guess for the solution vector x
    :return: four values: x_k (the solution vector), error (the error in the solution), iteraciones (the
    number of iterations performed), and elapsed_time (the time taken to execute the function).
    """
    alpha = 1
    omega = 1
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = x0
    Am = W + 1j*T
    b = p + 1j*q
    iteraciones = 0
    start_time = time.time()
    for k in range(max_iter):
        iteraciones = k+1
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
        error = np.linalg.norm(np.dot(Am,x_k)-b)

        if np.linalg.norm(np.dot(Am,x_k)-b) < np.linalg.norm(b)*tol:
            elapsed_time = time.time() - start_time
            return x_k, error, iteraciones, elapsed_time
    elapsed_time = time.time() - start_time
    return x_k, error, max_iter, elapsed_time
def PSHSS(W, T, p, q, alpha, omega, max_iter, tol, x0):
    """
    The function `PSHSS` solves a linear system of equations using the Preconditioned Hermitian and
    Skew-Symmetric (PSHSS) method.
    
    :param W: The parameter W represents a real-valued matrix
    :param T: The parameter T represents the imaginary part of the matrix A in the linear system Ax = b.
    It is a square matrix of size n x n
    :param p: The parameter `p` represents the real part of the right-hand side vector in the linear
    system of equations. It is a 1-dimensional numpy array of length `n`, where `n` is the size of the
    system
    :param q: The parameter `q` represents the imaginary part of the right-hand side vector in the
    linear system of equations
    :param alpha: Alpha is a scalar value used in the PSHSS algorithm. It is multiplied by the identity
    matrix to create the matrix I
    :param omega: Omega is a scalar parameter used in the PSHSS function. It is used to control the
    convergence behavior of the iterative solver
    :param max_iter: The parameter "max_iter" represents the maximum number of iterations that the
    algorithm will perform before terminating
    :param tol: The parameter "tol" stands for tolerance. It is a small positive value that determines
    the convergence criteria for the iterative solver. The algorithm will stop iterating when the norm
    of the residual (difference between the computed solution and the actual solution) is less than the
    product of the norm of the right-hand side
    :param x0: The initial guess for the solution vector x
    :return: the following values:
    1. x_k: The solution vector.
    2. error: The error between the product of matrix Am and x_k and the vector b.
    3. iteraciones: The number of iterations performed.
    4. elapsed_time: The time taken to execute the function.
    """
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = x0
    Am = W + 1j*T
    b = p + 1j*q

    iteraciones = 0
    start_time = time.time()
    for k in range(max_iter):
        iteraciones = k+1
        X = alpha * I + omega * W + T
        Z = (alpha * I - 1j * (omega * T - W)) @ x_k + \
            (omega - 1j) * (p + 1j * q)
        x_k = np.linalg.solve(X, Z)

        # Calcular el error
        error = np.linalg.norm(np.dot(Am,x_k)-b)

        if np.linalg.norm(np.dot(Am,x_k)-b) < np.linalg.norm(b)*tol:
            elapsed_time = time.time() - start_time
            return x_k, error, iteraciones,elapsed_time
    elapsed_time = time.time() - start_time
    return x_k, error, max_iter,elapsed_time

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
    algorithm will perform before stopping. It is used as a stopping criterion to prevent the algorithm
    from running indefinitely
    :param tol: The parameter "tol" stands for tolerance. It is a small positive number that determines
    the desired accuracy of the solution. The algorithm will stop iterating when the error, which is the
    difference between the actual solution and the approximation, is smaller than tol times the norm of
    the right-hand side vector A[
    :return: the final approximation of the solution vector x, the error of the approximation, the
    number of iterations performed, and the elapsed time in seconds.
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
    iterations = 0
    start_time = time.time()
    # Iteración principal del método
    for k in range(iter_max):
        iterations = k+1
        # Cálculo de la nueva aproximación x(k+1)
        x_new = M.dot(x) + N.dot(A[1])

        error = np.linalg.norm(A[0].dot(x_new) - A[1])
        # Comprobación del criterio de parada
        if np.linalg.norm(A[0].dot(x_new) - A[1]) < tol*np.linalg.norm(A[1]):
            elapsed_time = time.time() - start_time
            return x_new, error,iterations, elapsed_time
        x = x_new
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(A[0].dot(x_new) - A[1])
    return x,error, iter_max,elapsed_time

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
    The function `qr_factorization` performs QR factorization on a given matrix A and returns the
    matrices Q and R.
    
    :param A: A is a matrix of shape (m, n) that we want to factorize using QR factorization
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
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R
def solGaussiana(W,T,b):
    """
    The function `solGaussiana` solves a system of linear equations using Gaussian elimination and
    calculates the execution time and error of the solution.
    
    :param W: The parameter W represents the real part of the matrix in the system of equations
    :param T: The parameter T represents the imaginary part of the matrix W
    :param b: The parameter "b" represents a complex vector of constants in a system of linear equations
    :return: The function `solGaussiana` returns two values: `excecTime` and `error_gaussian`.
    """
    p = np.real(b)
    q = np.imag(b)
    # Construir la matriz aumentada y el vector de constantes
    A_real = np.block([[W, -T], [T, W]])
    d_real = np.concatenate([p, q])
    # Resolver usando eliminación gaussiana
    sol_gaussian = gaussian_elimination_complex(A_real, d_real)
    # Verificar las soluciones convirtiendo a números complejos
    A_complex = W + 1j * T
    
    startTime = time.time()
    x_gaussian = sol_gaussian[:len(sol_gaussian)//2] + 1j * sol_gaussian[len(sol_gaussian)//2:]
    #Calcular Ax y comparar con el lado derecho del sistema original
    Ax_gaussian = np.dot(A_complex, x_gaussian)

    excecTime = time.time()-startTime
    # Calcular errores
    error_gaussian = np.linalg.norm(Ax_gaussian - b)
    
    
    return excecTime,error_gaussian

def solQR(W,T,b):
    """
    The function `solQR` calculates the execution time and error of a QR factorization and solves a
    linear system using the factorization.
    
    :param W: The parameter W represents the real part of the matrix A in the equation Ax = b
    :param T: The parameter T represents the imaginary part of the matrix A. It is used in the
    calculation of the real and complex parts of the matrix A
    :param b: The parameter `b` is a complex-valued vector
    :return: The function `solQR` returns two values: `excecTime` and `error_qr_manual`.
    """
    p = np.real(b)
    q = np.imag(b)
    start_time = time.time()
    A_real = np.block([[W, -T], [T, W]])
    d_real = np.concatenate([p, q])
    A_complex = W + 1j * T
    # Resolver usando factorización QR sin np.linalg.qr
    Q_manual, R_manual = qr_factorization(A_real)
    sol_qr_manual = np.linalg.solve(R_manual, np.dot(Q_manual.T, d_real))
    x_qr_manual = sol_qr_manual[:len(sol_qr_manual)//2] + 1j * sol_qr_manual[len(sol_qr_manual)//2:]
    Ax_qr_manual = np.dot(A_complex, x_qr_manual)
    excecTime = time.time()-start_time
    error_qr_manual = np.linalg.norm(Ax_qr_manual - b)
    return excecTime, error_qr_manual

#generacion de sistemas lineales
linear_systems = []
##the limits of the range can be changed. right now it is from 2^4 to 2^6. If more testing is needed, the upper limit has to be changed, but it requires a lot of ram
for i in range(4,6):
    m = 2**i
    w,t,b = W_T_b(m)
    linear_systems.append((m,w,t,b))
    print(m)

##METODO 1 HSS
for i in range(0,len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    x, iter, elapsed_time, error = hss(A,b,x0,ITERMAX, TOL)
    print("Metodo1:\tHSS")
    print("\tCaso:",(i+1),"\t","m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\tIteraciones = ",iter)
    print("\n")

#METODO 2 PNHSS
for i in range(len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    x_k, error, iter, elapsed_time =PNHSS(W, T, np.real(b), np.imag(b),ITERMAX, TOL, x0)
    print("Metodo2:\tPNHSS")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\tIteraciones = ",iter)
    print("\n")

#METODO 3 PSHSS(W, T, p, q, alpha, omega, max_iter, tol, x0)for i in range(len(linear_systems)):
for i in range(len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    x_k, error, iter, elapsed_time =PSHSS(W, T, np.real(b), np.imag(b),1,1,ITERMAX, TOL, x0)
    print("Metodo3:\tPSHSS")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\tIteraciones = ",iter)
    print("\n")

#METODO 4 MHSS(W, T, p, q, alpha, omega, max_iter, tol, x0)for i in range(len(linear_systems)):
for i in range(len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    mhss((W+1j*T,b),x0,ITERMAX,TOL)
    
    x_k, error, iter, elapsed_time = mhss((W+1j*T,b),x0,ITERMAX,TOL)
    print("Metodo4:\tmHSS")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\tIteraciones = ",iter)
    print("\n")

#METODO 5 eliminacion gaussiana(W, T, p, q, alpha, omega, max_iter, tol, x0)for i in range(len(linear_systems)):
for i in range(len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    mhss((W+1j*T,b),x0,ITERMAX,TOL)
    
    elapsed_time,error = solGaussiana(W,T,b)
    print("Metodo5:\t eliminación gaussiana")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\n")

#METODO 6 factorizacion QR (W, T, p, q, alpha, omega, max_iter, tol, x0)for i in range(len(linear_systems)):
for i in range(len(linear_systems)):
    n = linear_systems[i][0]
    W = linear_systems[i][1]
    T = linear_systems[i][2]
    b = linear_systems[i][3]
    A = W + 1j*T
    x0 = np.zeros((n**2,1))
    mhss((W+1j*T,b),x0,ITERMAX,TOL)
    
    elapsed_time,error = solQR(W,T,b)
    print("Metodo6:\t factorizacion QR")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\n")