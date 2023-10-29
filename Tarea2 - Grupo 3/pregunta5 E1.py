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
print(b)
