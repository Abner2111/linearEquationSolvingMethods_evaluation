import numpy as np

def gaussian_elimination(M, d):
    # Perform Gaussian elimination
    n = len(d)
    augmented_matrix = np.concatenate((M.astype(np.float64), d.astype(np.float64).reshape(-1, 1)), axis=1)

    for i in range(n):
        # Pivot the matrix
        pivot_row = max(range(i, n), key=lambda k: abs(augmented_matrix[k, i]))
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Eliminate below the pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j, -1] -= augmented_matrix[j, i] * x[i]

    return x

def qr_factorization(M):
    # Perform QR factorization
    Q, R = np.linalg.qr(M.astype(np.float64))
    return Q, R

# Example usage
W = np.array([[1, 2], [3, 4]])
T = np.array([[5, 6], [7, 8]])
p = np.array([9, 10])
q = np.array([11, 12])

A = W + 1j * T
b = p + 1j * q

# Create the real-valued system
M = np.block([[W, -T], [T, W]])
d = np.concatenate([p, q])

# Solve using Gaussian elimination
solution_gaussian = gaussian_elimination(M, d)
print("Solution (Gaussian elimination):", solution_gaussian)

# Solve using QR factorization
Q, R = qr_factorization(M)
solution_qr = np.linalg.solve(R, np.dot(Q.T, d))
print("Solution (QR factorization):", solution_qr)

# Verify the solution
Ax = np.dot(A, solution_gaussian[:2])  # Adjust indices based on the size of the matrices
error = np.linalg.norm(Ax - b, 2)
print("Error:", error)
