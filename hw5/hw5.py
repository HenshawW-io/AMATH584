import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os

'''QUESTION 1'''
m = 10
num_iterations = 25
# A = np.array([[0.5, 0.5], [0.2, 0.8]])
# A = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
A = np.random.randn(m,m)
A_symm = ( (A + A.T) / 2 )
cond = np.linalg.cond(A_symm)

# while cond < 10e-16:
#     A = np.random.randn(m,m)
#     A_symm = ( (A + A.T) / 2 )
#     cond = la.cond(A_symm)

approx_vals = []
approx_vals_real = []
approx_vals_imag = []
error_vecs = []
error_vals = []
# print(A_symm)

eigvals_truth, eigvecs_truth = la.eig(A_symm)
eigvals_truth = np.abs(eigvals_truth)

idx = eigvals_truth.argsort()[::-1]
eigvals_truth = eigvals_truth[idx]
eigvecs_truth = eigvecs_truth[:, idx]

# print(eigvals_truth)
# print(eigvecs_truth)
max_eigval_truth = eigvals_truth[0]
max_eigvec_truth = eigvecs_truth[:, 0]

# Convergence ratio. If equal to one, power iteration will not converge.
convergence_ratio = np.abs(eigvals_truth[1] / eigvals_truth[0])**2
# convergence_ratio = np.abs(eigvals_truth[1] / eigvals_truth[0])

if convergence_ratio > 0.98:
    print('Does not converge!')
    # raise('Input Error')
else:
    print('\nConvergence ratio\n----------------------\n', convergence_ratio)

# Power iteration algorithm

# "first guess" eigenvector
# v = np.random.randn(A_symm.shape[1])
v0 = np.random.rand(A_symm.shape[1])
v = v0
'''
# normalize first guess
v = v / la.norm(v)

for _ in range(num_iterations):

    # calculating next eigenvector approximation
    v_next = A @ v # (m,1)

    # calculate norm
    v_next_norm = la.norm(v_next, 2)

    # normalizing eigenvector approximation
    v = v_next / v_next_norm # (m,1)
    max_eigval_approx = v.T @ ( A @ v ) # (1,m) * (m,m) * (m,1) = (1,1)

    # print(max_eigval_approx)
    approx_vals.append(max_eigval_approx)
    approx_vals_real.append(max_eigval_approx.real)
    approx_vals_imag.append(max_eigval_approx.imag)
    # error_vecs.append(la.norm(np.abs(v) - np.abs(max_eigvec_truth), 2))
    # print(max_eigval_approx)
    # print(la.norm(v - max_eigvec_truth, 2))
    error_vals.append(np.abs(max_eigval_approx) - np.abs(max_eigval_truth))

# print('\nGround truth max eigvec\n----------------------\n', max_eigvec_truth)
# print(f'\nApproximate max eigvec after {num_iterations} iterations\n----------------------\n', v)

print('\nGround truth max eigval\n----------------------\n', max_eigval_truth)
print(f'\nApproximate max eigval after {num_iterations} iterations\n----------------------\n', max_eigval_approx)


# print(max_eig_approx)

# X,Y = np.meshgrid(m,m)
# plt.plot(error_vecs)
# # plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Error')
# plt.show()

# plt.scatter(approx_vals_real, approx_vals_imag)
# plt.show()
#
plt.plot(error_vals)
# plt.yscale('log')
plt.title('Symmetric A, dominant eigenvalue approximation error: power iteration')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.savefig(os.getcwd() + '/hw5q1pb.png', bbox_inches='tight')
plt.show()
'''
'''Rayleigh Quotient iteration'''
err_thresh = 10e-8 # error threshold for eigenvalue approx

print('\nA = ')
print(A_symm)

# v1 = v0 + 4
# v2 = v0 + 6
for k in range(m):
    v = eigvecs_truth[k] # set initial eigenvector guess
    lambda0 = v.T @ A_symm @ v # first eigenvalue approximation

    lam = lambda0 # set initial eigenvalue guess
    eigval_guesses = [lambda0]
    eigvec_guesses = [v]
    # realizations = 10
    # for i in range(m): # isolating different direction of initial guess vector
    #     v0 = np.zeros(A_symm.shape[1])
    #     v0[i] = 1
    # v = v0 # set initial eigenvector guess

    print('\n Eigenvectors: ', eigvecs_truth)
    print('\nStarting guess v: ', v)
    print(' ')

    print('\nEigenvalues: ', eigvals_truth)
    print('\nStarting guess lambda: ', lambda0)
    print(' ')

    print(f'\ni = 0, lambda = {lambda0}')

    for i in range(1, num_iterations):
        B = A_symm - lam*np.eye(m)
        try:
            omega = la.solve(B, v)
        except:
            print("Matrix is singular! Converged solution")
            break
        v = omega / la.norm(omega, 2)
        lam = v.T @ ( A_symm @ v )
        print(f'i = {i}, lambda = {lam}')
        eigval_guesses.append(lam)
        eigvec_guesses.append(v)
        error = np.abs(eigval_guesses[-1] - eigval_guesses[-2])
        # print(error)
        # if error < err_thresh:
        #     break

    print('\nEigenvalue approximation: ', eigval_guesses[-1])
    plt.plot(eigval_guesses)
    # plt.yscale('log')
    plt.title(f'Eigenvalue {k+1} approximation: Rayleigh Quotient iteration', fontsize=10)
    plt.xlabel('Iteration')
    plt.ylabel('Guess')
    # plt.savefig(os.getcwd() + '/hw5q1pc.png', bbox_inches='tight')
    plt.show()
