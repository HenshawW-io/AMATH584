import numpy as np
import math
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
import os

'''QUESTION 1: Non-symmetric'''

def complex_modulo(z):
    a = z.real
    b = z.imag
    return math.sqrt(a**2 + b**2)

m = 10
num_iterations = 100
A = np.random.randn(m,m)
# A_symm = ( (A + A.T) / 2 )
cond = np.linalg.cond(A)

# while cond < 10e-16:
#     A = np.random.randn(m,m)
#     A_symm = ( (A + A.T) / 2 )
#     cond = la.cond(A_symm)

approx_vals = []
approx_vals_real = []
approx_vals_imag = []
error_vecs = []
error_vals = []
eigvals_truth_modulo = []

eigvals_truth, eigvecs_truth = la.eig(A)
for i in eigvals_truth:
    eigvals_truth_modulo.append(complex_modulo(i))

idx = eigvals_truth.argsort()[::-1]
eigvals_truth = eigvals_truth[idx]
eigvecs_truth = eigvecs_truth[:, idx]
print('\nEigenvalues: ', eigvals_truth)

max_eigval_truth = np.max(eigvals_truth_modulo)
# max_eigvec_truth = eigvecs_truth[:, 0]

# Convergence ratio. If equal to one, power iteration will not converge.
convergence_ratio = np.abs(eigvals_truth[1] / eigvals_truth[0])

if convergence_ratio > 0.98:
    print('Does not converge!')
    exit()
    # raise('Input Error')
else:
    print('\nConvergence ratio\n----------------------\n', convergence_ratio)

# Power iteration algorithm

# "first guess" eigenvector
# v = np.random.randn(A_symm.shape[1])
v0 = np.random.rand(A.shape[1]) + np.random.rand(A.shape[1]) * 1j
v = v0

max_eigval_approx = v.T @ ( A @ v ) # (1,m) * (m,m) * (m,1) = (1,1)
approx_vals.append(max_eigval_approx)


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
    approx_vals.append(complex_modulo(max_eigval_approx))
    approx_vals_real.append(max_eigval_approx.real)
    approx_vals_imag.append(max_eigval_approx.imag)
    # error_vecs.append(la.norm(np.abs(v) - np.abs(max_eigvec_truth), 2))
    # print(max_eigval_approx)
    # print(la.norm(v - max_eigvec_truth, 2))
    error_vals.append(approx_vals[-1] - max_eigval_truth)

# print('\nGround truth max eigvec\n----------------------\n', max_eigvec_truth)
# print(f'\nApproximate max eigvec after {num_iterations} iterations\n----------------------\n', v)

# print('\nGround truth max eigval\n----------------------\n', max_eigval_truth)
print(f'\nApproximate max eigval after {num_iterations} iterations\n----------------------\n', max_eigval_approx)


# print(max_eig_approx)
#
# X,Y = np.meshgrid(m,m)
# plt.plot(error_vecs)
# # plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Error')
# plt.show()

# plt.scatter(approx_vals_real, approx_vals_imag)
# plt.title('Non-symmetric A, dominant eigenvalue approximation error: power iteration', fontsize=10)
# plt.show()
#

plt.scatter(approx_vals_real, approx_vals_imag)
# plt.yscale('log')
plt.title('Non-symmetric A, dominant eigenvalue approximation: power iteration', fontsize=10)
plt.xlabel('Real Part')
plt.ylabel('Imag Part')
# plt.savefig(os.getcwd() + '/hw5q1pd_nonsymm.png', bbox_inches='tight')
plt.show()

plt.plot(error_vals)
# plt.yscale('log')
plt.title('Non-symmetric A, dominant eigenvalue approximation: power iteration', fontsize=10)
plt.xlabel('Iteration')
plt.ylabel('Error')
# plt.savefig(os.getcwd() + '/hw5q1pd_nonsymm.png', bbox_inches='tight')
plt.show()


'''Rayleigh Quotient iteration'''
err_thresh = 10e-8 # error threshold for eigenvalue approx

print('\nA = ')
print(A)

rqiter_eigval_approx = []
rqiter_eigval_approx_real = []
rqiter_eigval_approx_imag = []

# v1 = v0 + 4
# v2 = v0 + 6

def allUnique(x): # checks for uniqueness
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

timeout = time.time() + 60*5   # 5 minutes from now

while len(rqiter_eigval_approx) < m:
    while allUnique(rqiter_eigval_approx) == True:

        if time.time() > timeout:
            break

        v = np.random.rand(A.shape[1]) + np.random.rand(A.shape[1]) * 1j # set initial eigenvector guess
        lambda0 = v.T @ A @ v # first eigenvalue approximation

        lam = lambda0 # set initial eigenvalue guess
        eigval_guesses = [lam]
        eigval_guesses_real = [lam.real]
        eigval_guesses_imag = [lam.imag]
        eigvec_guesses = [v]
        # realizations = 10
        # for i in range(m): # isolating different direction of initial guess vector
        #     v0 = np.zeros(A_symm.shape[1])
        #     v0[i] = 1
        # v = v0 # set initial eigenvector guess

        # print('\n Eigenvectors: ', eigvecs_truth)
        # print('\nStarting guess v: ', v)
        # print(' ')

        # print('\nEigenvalues: ', eigvals_truth)
        print('\nStarting guess lambda: ', lambda0)
        print(' ')

        # print(f'\ni = 0, lambda = {lambda0}')

        for i in range(1, num_iterations):
            B = A - lam*np.eye(m)
            try:
                omega = la.solve(B, v)
            except:
                print("Matrix is singular! Converged solution")
                break

            v = omega / la.norm(omega, 2)
            lam = v.T @ ( A @ v )
            # print(f'i = {i}, lambda = {lam}')
            eigval_guesses.append(lam)
            eigval_guesses_real.append(lam.real)
            eigval_guesses_imag.append(lam.imag)
            eigvec_guesses.append(v)
            # error = np.abs(eigval_guesses[-1] - eigval_guesses[-2])

        rqiter_eigval_approx.append(np.round(complex_modulo(eigval_guesses[-1]), 4))
        print(rqiter_eigval_approx[-1])
        rqiter_eigval_approx_real.append(np.round(np.abs(eigval_guesses_real[-1]), 8))
        rqiter_eigval_approx_imag.append(np.round(np.abs(eigval_guesses_imag[-1]), 8))
        print('\nEigenvalue approximation: ', eigval_guesses[-1])

        if allUnique(rqiter_eigval_approx) == False:
            del rqiter_eigval_approx[-1]
        else:
            plt.scatter(rqiter_eigval_approx_real[-1], rqiter_eigval_approx_imag[-1])
            # plt.yscale('log')
            plt.title(f'Non-symmetric A, Eigenvalue approximation: Rayleigh Quotient iteration', fontsize=10)
            plt.xlabel('Real')
            plt.ylabel('Imag')
            # plt.savefig(os.getcwd() + f'/hw5q1pc{len(rqiter_eigval_approx)}_nonsymm.png', bbox_inches='tight')
            # plt.close()

print('\nEigenvalues: ', eigvals_truth)
print('\n Eigenvalue Approximations: ', rqiter_eigval_approx)
plt.show()
