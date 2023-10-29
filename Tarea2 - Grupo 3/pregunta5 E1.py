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
        if np.linalg.norm(np.dot(A, x) - b, 2) <= np.linalg.norm(b)*tol:
            # Cálculo del tiempo de ejecución
            elapsed_time = time.time() - start_time

            # Cálculo del error en comparación con la solución exacta
            error = np.linalg.norm(np.dot(A, x) - b, 2)
            

            # Devolver variables por separado
            return x, iteraciones, elapsed_time, error

    # Si no converge en max_iter iteraciones
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(np.dot(A, x) - b, 2)
    return x, max_iter, elapsed_time, error


def PNHSS(W, T, p, q,max_iter, tol, x0):
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
    # Tamaño del vector x0
    m = len(x0)
    # Matriz identidad del mismo tamaño que x0
    Im = np.eye(m, dtype=complex)
    x = x0

    # Descomposición de la matriz A en W y T
    W, T = A
    # Cálculo de los valores propios de la matriz W
    eigenvalues_W = np.linalg.eigvals(W)
    # Cálculo de alpha* según la fórmula dada
    alpha_star = np.sqrt(np.min(eigenvalues_W) * np.max(eigenvalues_W))
    # Cálculo de M(α) y N(α) según las fórmulas
    M = np.linalg.inv(alpha_star * Im + T).dot((alpha_star * Im + 1j * W).dot(np.linalg.inv(alpha_star * Im + W).dot(alpha_star * Im - 1j * T)))
    N = (1 - 1j) * alpha_star * np.linalg.inv(alpha_star * Im + T).dot(np.linalg.inv(alpha_star * Im + W))

    # Iteración principal del método
    for k in range(iter_max):
        # Cálculo de la nueva aproximación x(k+1)
        x_new = M.dot(x) + N.dot(A[1])
        # Comprobación del criterio de parada
        if np.linalg.norm(A[0].dot(x_new) - A[1]) < tol*np.linalg.norm(A[1]):
            return x_new
        x = x_new

    return x

def gaussian_elimination_complex(M, d):
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

#generacion de sistemas lineales
linear_systems = []

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