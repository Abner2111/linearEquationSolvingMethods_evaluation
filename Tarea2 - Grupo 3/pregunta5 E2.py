import numpy as np

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

