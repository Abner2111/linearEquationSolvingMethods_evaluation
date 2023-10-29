import numpy as np
import time
ITERMAX = 5000
TOL = 1e-6
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

def v(m,h):
    Vm = (1/h**2)*tridiagMatrix(-1,2,-1,m)
    
    return Vm

def k(m, I,h):
    Vm = v(m,h)
    K = np.kron(I,Vm) + np.kron(Vm,I)
    return K
def b_(A):
    one = np.ones((len(A), 1))
    B = (1+1j)*np.dot(A,one)
    return B
def W_T_b(m):
    sigma1 = 1
    sigma2 = 1
    h = 1/(m+1)
    Im = np.identity(m)
    Imm = np.identity(m**2)
    K = k(m, Im, h)
    W = K + np.dot(sigma1,Imm)
    T = sigma2*Imm
    B = b_(W+1j*T)
    

    return W*(h**2),T*(h**2),B*(h**2)
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
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R
def solGaussiana(W,T,b):
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
    x_gaussian = sol_gaussian[:2] + 1j * sol_gaussian[2:]
    # Calcular Ax y comparar con el lado derecho del sistema original
    Ax_gaussian = np.dot(A_complex, x_gaussian)

    endTime = time.time()
    # Calcular errores
    error_gaussian = np.linalg.norm(Ax_gaussian - b)
    excecTime = endTime-startTime
    return excecTime,error_gaussian

def solQR(W,T,b):
    p = np.real(b)
    q = np.imag(b)
    start_time = time.time()
    A_real = np.block([[W, -T], [T, W]])
    d_real = np.concatenate([p, q])
    A_complex = W + 1j * T
    # Resolver usando factorización QR sin np.linalg.qr
    Q_manual, R_manual = qr_factorization(A_real)
    sol_qr_manual = np.linalg.solve(R_manual, np.dot(Q_manual.T, d_real))
    x_qr_manual = sol_qr_manual[:2] + 1j * sol_qr_manual[2:]
    Ax_qr_manual = np.dot(A_complex, x_qr_manual)
    excecTime = time.time()-start_time
    error_qr_manual = np.linalg.norm(Ax_qr_manual - b)
    return excecTime, error_qr_manual

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
    print("Metodo5:\t factorizacion QR")
    print("\tCaso:",(i+1),"\t"," m=",n)
    print("\terror = ",error)
    print("\tTiempo de ejecucion = ",elapsed_time," segs")
    print("\n")