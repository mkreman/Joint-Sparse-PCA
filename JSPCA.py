import numpy as np
from numpy.linalg import inv, norm, svd, eig
import scipy.io



def orthogonal(A):
    B = np.zeros_like(A)
    for i in range(A.shape[1]):
        vec = A[:, i]
        for j in range(i):
            vec -= np.dot(A[:, i], B[:, j]) * B[:, j]
        B[:, i] = vec / norm(vec)
    return B
    

def jspca(X: np.array, lambda_: float, maxiter: int, d: int) -> np.array:
    """
    :param X: Input array
    :param lambda_: Regularization parameter
    :param maxiter: Maximum iterations
    :return:
    """
    m, n = X.shape
    # Initializing D1, D2, P1
    D1 = np.eye(m)
    D2 = np.eye(m)
    P = orthogonal(np.random.randn(m, d))
    for _ in range(maxiter):
        Q = inv(lambda_ * D2 + X @ X.T) @ X @ X.T @ np.sqrt(D1) @ P
        E, D, U = svd(np.sqrt(D1) @ X @ X.T @ np.sqrt(D1) @ Q, full_matrices=False)
        D = np.diag(D)

        P = E @ U.T
        A = X - P @ Q.T @ X
        middle1 = np.sqrt((A ** 2).sum(axis=1))
        middle1[middle1 == 0] = 1e-6
        D1 = np.diag(1 / (2 * middle1))

        middle2 = np.sqrt((Q ** 2).sum(axis=1))
        middle2[middle2 == 0] = 1e-6 # For zero entries add epsilon
        D2 = np.diag(1 / (2 * middle2))  

        Q_norm = np.sqrt((np.square(Q)).sum(axis=0))
        Q_norm[Q_norm == 0] = 1
        for i in range(d):
            Q[:, i] /= Q_norm[i]
    return P, Q


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(10, 11)
    print(jspca(X, lambda_=0.01, maxiter=10, d=6))
