import numpy as np

W = np.array([[1, 0], [0, 2]])  # Matriz W
T = np.array([[2, 0], [0, 3]])  # Matriz T
b = np.array([1, 2], dtype=complex)  # Vector b
alpha = 9  
omega = 1  
max_iter = 100  
tol = 1e-6 

 
#                     __________________________________________________
#___________________/Efficient parameterized HSS iteration (algorithm 2)

def PNHSS(W, T, b, alpha, omega, max_iter, tol ): 
    
    I = np.eye(1) * alpha
    x = np.zeros(len(b), dtype=complex)
    
    for k in range(max_iter):
        # Primer paso
        A = -1j * np.dot(omega * T - W, x) + np.dot(omega - 1j, b)
        print("Solución A:", A)
        # Segundo paso
        B = np.dot(alpha * I - 1j*(alpha*T - W), x) + (omega - 1j) * b
        print("Solución B:", B)

        # Actualizar x
        x = np.linalg.solve(B, A)

        # Calcular el error
        error = np.linalg.norm(A - np.dot(B, x))
        
        if error < tol:
            return x
        
    return x


solution = PNHSS(W, T, b, alpha, omega, max_iter, tol)
print("Solución:", solution)


#                     __________________________________________________
#___________________/Efficient parameterized HSS iteration (algorithm 3)
#def PSHSS(q): 
