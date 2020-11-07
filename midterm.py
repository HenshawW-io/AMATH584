import scipy.linalg as la
import numpy as np

print('\n')

m = 4
# n = 4

A = np.random.rand(m, m); U = np.copy(A); L = np.identity(m); P = np.identity(m)
# print(A)

# i: row indx
# k: row iterator

for k in range(m-1):
    i = np.argmax(np.abs(U[k:, k])) + k
    U[[i, k]] = U[[k, i]]
    L[[i, k], :k] = L[[k, i], :k]
    P[[i, k], :] = P[[k, i], :]
    for j in range(k+1, m):
        L[j, k] = U[j, k] / U[k, k]
        U[j, k:] = U[j, k:] - ( L[j, k] * U[k, k:] )

print('A = \n', A, '\n\n')
print('P = \n', P, '\n\n')
print('L = \n', L, '\n\n')
print('U = \n', U, '\n\n')

print('PA = \n', np.matmul(P, A), '\n\n', 'LU = \n', np.matmul(L, U), '\n\n')
