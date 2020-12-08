import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
import os

'''QUESTION 1'''
m = 10
num_iterations = 250
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

eigvals_truth, eigvecs_truth = la.eig(A_symm)
eigvals_truth = np.abs(eigvals_truth)

idx = eigvals_truth.argsort()[::-1]
eigvals_truth = eigvals_truth[idx]
eigvecs_truth = eigvecs_truth[:, idx]

max_eigval_truth = eigvals_truth[0]
max_eigvec_truth = eigvecs_truth[:, 0]

# Convergence ratio. If equal to one, power iteration will not converge.
convergence_ratio = np.abs(eigvals_truth[1] / eigvals_truth[0])**2
# convergence_ratio = np.abs(eigvals_truth[1] / eigvals_truth[0])

if convergence_ratio > 0.98:
    print('Does not converge!')
    exit()
    # raise('Input Error')
else:
    print('\nConvergence ratio\n----------------------\n', convergence_ratio)

# Power iteration algorithm

# "first guess" eigenvector
# v = np.random.randn(A_symm.shape[1])
v0 = np.random.rand(A.shape[1])
v = v0

max_eigval_approx = v.T @ ( A_symm @ v ) # (1,m) * (m,m) * (m,1) = (1,1)
approx_vals.append(max_eigval_approx)


# normalize first guess
v = v / la.norm(v)

for _ in range(num_iterations):

    # calculating next eigenvector approximation
    v_next = A_symm @ v # (m,1)

    # calculate norm
    v_next_norm = la.norm(v_next, 2)

    # normalizing eigenvector approximation
    v = v_next / v_next_norm # (m,1)
    max_eigval_approx = v.T @ ( A_symm @ v ) # (1,m) * (m,m) * (m,1) = (1,1)

    # print(max_eigval_approx)
    approx_vals.append(max_eigval_approx)
    approx_vals_real.append(max_eigval_approx.real)
    approx_vals_imag.append(max_eigval_approx.imag)
    # error_vecs.append(la.norm(np.abs(v) - np.abs(max_eigvec_truth), 2))
    # print(max_eigval_approx)
    # print(la.norm(v - max_eigvec_truth, 2))
    # error_vals.append(np.abs(max_eigval_approx) - np.abs(max_eigval_truth))

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
plt.plot(np.abs(approx_vals) - np.abs(max_eigval_truth))
# plt.yscale('log')
plt.title('Symmetric A, dominant eigenvalue approximation error: power iteration', fontsize=10)
plt.xlabel('Iteration')
plt.ylabel('Error')
# plt.savefig(os.getcwd() + '/hw5q1pd.png', bbox_inches='tight')
plt.show()

'''Rayleigh Quotient iteration'''
err_thresh = 10e-8 # error threshold for eigenvalue approx

print('\nA = ')
print(A_symm)

rqiter_eigval_approx = []

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

        v = np.random.randn(A_symm.shape[1]) # set initial eigenvector guess
        lambda0 = v.T @ A_symm @ v # first eigenvalue approximation

        lam = lambda0 # set initial eigenvalue guess
        eigval_guesses = [lam]
        eigvec_guesses = [v]
        # realizations = 10
        # for i in range(m): # isolating different direction of initial guess vector
        #     v0 = np.zeros(A_symm.shape[1])
        #     v0[i] = 1
        # v = v0 # set initial eigenvector guess

        # print('\n Eigenvectors: ', eigvecs_truth)
        # print('\nStarting guess v: ', v)
        # print(' ')

        print('\nEigenvalues: ', eigvals_truth)
        print('\nStarting guess lambda: ', lambda0)
        print(' ')

        # print(f'\ni = 0, lambda = {lambda0}')

        for i in range(1, num_iterations):
            B = A_symm - lam*np.eye(m)
            try:
                omega = la.solve(B, v)
            except:
                print("Matrix is singular! Converged solution")
                break

            v = omega / la.norm(omega, 2)
            lam = v.T @ ( A_symm @ v )
            # print(f'i = {i}, lambda = {lam}')
            eigval_guesses.append(lam)
            eigvec_guesses.append(v)
            error = np.abs(eigval_guesses[-1] - eigval_guesses[-2])

        rqiter_eigval_approx.append(np.round(np.abs(eigval_guesses[-1]), 8))
        print('\nEigenvalue approximation: ', eigval_guesses[-1])

        if allUnique(rqiter_eigval_approx) == False:
            del rqiter_eigval_approx[-1]
        else:
            plt.plot(eigval_guesses)
            # plt.yscale('log')
            plt.title(f'Eigenvalue approximation: Rayleigh Quotient iteration', fontsize=10)
            plt.xlabel('Iteration')
            plt.ylabel('Guess')
            plt.savefig(os.getcwd() + f'/hw5q1pc{len(rqiter_eigval_approx)}.png', bbox_inches='tight')
            plt.show

print('\nEigenvalues: ', eigvals_truth)
print('\n Eigenvalue Approximations: ', rqiter_eigval_approx)
