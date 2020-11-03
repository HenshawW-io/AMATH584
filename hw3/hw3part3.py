import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
import scipy.io as sio

for i in range(5):
    m = 4
    n = 2
    cond = []
    while m < 100:
        A = np.random.randn(m,n)
        cond.append(np.linalg.cond(A))
        m += 1
        n += 1
    plt.plot(range(4, 100), cond)
    plt.title(f'Example Matrix {i+1}')
    plt.xlabel('m (n+2)')
    plt.ylabel('cond(A)')
    plt.yscale('log')
    plt.savefig(f'hw3part3plot{i+1}.png', bbox_inches='tight')
    plt.show()

# what about the relationship between cond and size with different m:n ratio?

cond_new = np.empty((100,100))
for i in range(cond_new.shape[0]):
    for j in range(cond_new.shape[1]):
        A = np.random.randn(i+1, j+1)
        cond_new[i, j] = np.linalg.cond(A)

a = plt.pcolormesh(cond_new, cmap='jet')
plt.gca().invert_yaxis()
plt.colorbar(a)
plt.savefig(f'hw3part3plot6.png', bbox_inches='tight')
plt.show()
# now let's fix m and n; copy the first column of A and append it as the (n+1)th
# columm. What is the determinant and cond number of the matrix?

def noise(epsil):
    return epsil * np.random.rand(m, 1)

m = 4
n = 3

A = np.random.randn(m,n)
A_new = np.hstack((A, np.atleast_2d(A[:,0]).T))
print('\n\n', A_new, '\n\n\n', "Determinant: ", np.linalg.det(A_new), '\n\n', \
"Condition Number: ", np.linalg.cond(A_new), '\n\n')

# what if we add noise to the last column of the matrix? What happens to the Condition
# Number if we change epsilon?

cond = []
# A_new[:, -1] += noise(1).flatten()
for i in range(100):
    A_new[:, -1] += noise(i).flatten()
    cond.append(np.linalg.cond(A_new))

# print(np.linalg.cond(A_new))
plt.plot(range(100), cond)
plt.xlabel('epsilon')
plt.ylabel('cond(A)')
plt.title('Part III, Question C')
plt.yscale('log')
plt.savefig(f'hw3part3qC.png', bbox_inches='tight')
plt.show()
