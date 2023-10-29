import time
import numpy as np
ITERMAX = 5000

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


w,t,b = W_T_b(3)
print(w)
print(t)
print(b)

def hss(A, b, x0, exact_solution, max_iter=1000, tol=1e-12):
    # Inicialización de variables
    n = A.shape[0]
    x = x0

    # Inicialización de variables para medir el tiempo
    start_time = time.time()

    # Iteraciones del método HSS
    for k in range(max_iter):
        # Cálculo del residuo
        r = b - np.dot(A, x)

        # Cálculo de la corrección z mediante la resolución de Ax = r
        z = np.linalg.solve(A, r)

        # Actualización de x con la corrección z
        x = x + z

        # Verificación de convergencia utilizando la norma Euclidiana
        if np.linalg.norm(np.dot(A, x) - b, 2) <= np.linalg.norm(b)*tol:
            # Cálculo del tiempo de ejecución
            elapsed_time = time.time() - start_time

            # Cálculo del error en comparación con la solución exacta
            error = np.linalg.norm(x - exact_solution, 2)
            

            # Devolver variables por separado
            return x, k + 1, elapsed_time, error, exact_solution

    # Si no converge en max_iter iteraciones
    elapsed_time = time.time() - start_time
    error = np.linalg.norm(x - exact_solution, 2)
    return x, max_iter, elapsed_time, error, exact_solution


def PNHSS(W, T, p, q,max_iter, tol, x0):
    alpha = 1
    omega = 1
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = np.zeros(n, dtype=complex)
    Am = W + 1j*T
    b = p + 1j*q
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

        if np.linalg.norm(np.dot(Am,n)-b) < np.linalg.norm(b)*tol:
            return x_k, error, max_iter

    return x_k, error, max_iter
def PSHSS(W, T, p, q, alpha, omega, max_iter, tol, x0):
    n = len(p)
    I = alpha * np.eye(n, dtype=complex)
    x_half = np.zeros(n, dtype=complex)
    x_k = np.zeros(n, dtype=complex)
    Am = W + 1j*T
    b = p + 1j*q

    for k in range(max_iter):
        X = alpha * I + omega * W + T
        Z = (alpha * I - 1j * (omega * T - W)) @ x_k + \
            (omega - 1j) * (p + 1j * q)
        x_k = np.linalg.solve(X, Z)

        # Calcular el error
        error = np.linalg.norm(x_k - x_half)

        if np.linalg.norm(np.dot(Am,n)-b) < np.linalg.norm(b)*tol:
            return x_k, error, max_iter

    return x_k, error, max_iter
