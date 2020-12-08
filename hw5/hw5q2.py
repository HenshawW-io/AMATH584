import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import glob
import cv2
import os
from PIL import Image

no_imgs = 64
num_iterations = 20

b01s = []
b02s = []
b03s = []
b04s = []
b05s = []
b06s = []
b07s = []
b08s = []
b09s = []
b10s = []
b11s = []
b12s = []
b13s = []
b14s = []
b15s = []
b16s = []
b17s = []
b18s = []
b19s = []
b20s = []
b21s = []
b22s = []
b23s = []
b24s = []
b25s = []
b26s = []
b27s = []
b28s = []
b29s = []
b30s = []
b31s = []
b32s = []
b33s = []
b34s = []
b35s = []
b36s = []
b37s = []
b38s = []
b39s = []

# print(cv2.__version__)
wd = '/home/billyh/Documents/AMATH 584: Linear Algebra/hw2_data/'

cropped_imgs = glob.glob(wd + 'cropped/CroppedYale/*/*')

'''This categorizes the images into their subjects'''
b01s = [cv2.imread(a, 0) for a in cropped_imgs if 'B01' in a]
b02s = [cv2.imread(a, 0) for a in cropped_imgs if 'B02' in a]
b03s = [cv2.imread(a, 0) for a in cropped_imgs if 'B03' in a]
b04s = [cv2.imread(a, 0) for a in cropped_imgs if 'B04' in a]
b05s = [cv2.imread(a, 0) for a in cropped_imgs if 'B05' in a]
b06s = [cv2.imread(a, 0) for a in cropped_imgs if 'B06' in a]
b07s = [cv2.imread(a, 0) for a in cropped_imgs if 'B07' in a]
b08s = [cv2.imread(a, 0) for a in cropped_imgs if 'B08' in a]
b09s = [cv2.imread(a, 0) for a in cropped_imgs if 'B09' in a]
b10s = [cv2.imread(a, 0) for a in cropped_imgs if 'B10' in a]
b11s = [cv2.imread(a, 0) for a in cropped_imgs if 'B11' in a]
b12s = [cv2.imread(a, 0) for a in cropped_imgs if 'B12' in a]
b13s = [cv2.imread(a, 0) for a in cropped_imgs if 'B13' in a]
b15s = [cv2.imread(a, 0) for a in cropped_imgs if 'B15' in a]
b16s = [cv2.imread(a, 0) for a in cropped_imgs if 'B16' in a]
b17s = [cv2.imread(a, 0) for a in cropped_imgs if 'B17' in a]
b18s = [cv2.imread(a, 0) for a in cropped_imgs if 'B18' in a]
b19s = [cv2.imread(a, 0) for a in cropped_imgs if 'B19' in a]
b20s = [cv2.imread(a, 0) for a in cropped_imgs if 'B20' in a]
b21s = [cv2.imread(a, 0) for a in cropped_imgs if 'B21' in a]
b22s = [cv2.imread(a, 0) for a in cropped_imgs if 'B22' in a]
b23s = [cv2.imread(a, 0) for a in cropped_imgs if 'B23' in a]
b24s = [cv2.imread(a, 0) for a in cropped_imgs if 'B24' in a]
b25s = [cv2.imread(a, 0) for a in cropped_imgs if 'B25' in a]
b26s = [cv2.imread(a, 0) for a in cropped_imgs if 'B26' in a]
b27s = [cv2.imread(a, 0) for a in cropped_imgs if 'B27' in a]
b28s = [cv2.imread(a, 0) for a in cropped_imgs if 'B28' in a]
b29s = [cv2.imread(a, 0) for a in cropped_imgs if 'B29' in a]
b30s = [cv2.imread(a, 0) for a in cropped_imgs if 'B30' in a]
b31s = [cv2.imread(a, 0) for a in cropped_imgs if 'B31' in a]
b32s = [cv2.imread(a, 0) for a in cropped_imgs if 'B32' in a]
b33s = [cv2.imread(a, 0) for a in cropped_imgs if 'B33' in a]
b34s = [cv2.imread(a, 0) for a in cropped_imgs if 'B34' in a]
b35s = [cv2.imread(a, 0) for a in cropped_imgs if 'B35' in a]
b36s = [cv2.imread(a, 0) for a in cropped_imgs if 'B36' in a]
b37s = [cv2.imread(a, 0) for a in cropped_imgs if 'B37' in a]
b38s = [cv2.imread(a, 0) for a in cropped_imgs if 'B38' in a]
b39s = [cv2.imread(a, 0) for a in cropped_imgs if 'B39' in a]

res = b01s[0].shape

matrix = np.stack([
b01s,
b02s,
b03s,
b04s,
b05s,
b06s,
b07s,
b08s,
b09s,
b10s,
b11s,
b12s,
b13s,
b15s,
b16s,
b17s,
b18s,
b19s,
b20s,
b21s,
b22s,
b23s,
b24s,
b25s,
b26s,
b27s,
b28s,
b29s,
b30s,
b31s,
b32s,
b33s,
b34s,
b35s,
b36s,
b37s,
b38s,
b39s])

matrix = matrix.reshape(matrix.shape[0] * matrix.shape[1], matrix.shape[2] * matrix.shape[3])
print(matrix.shape)

#
# '''Getting the average face for each subject'''
# cropped_subs = [b01s, b02s, b03s, b04s, b05s, b06s, b07s, b08s, b09s, b10s, b11s, b12s, b13s, b15s, b16s, b17s, b18s, b19s, b20s, b21s, b22s, b23s, b24s, b25s, b26s, b27s, b28s, b29s, b30s, b31s, b32s, b33s, b34s, b35s, b36s, b37s, b38s, b39s]
# averaged_cropped = []
# #
# for i in cropped_subs:
#     # print(i)
#     averaged_cropped.append(np.mean(i, 0))
#
# averaged_cropped_vec = np.stack(averaged_cropped, axis=2)
# # reshape images into columns
# averaged_cropped_vec = averaged_cropped_vec.reshape(res[0] * res[1], 38)
# print(averaged_cropped_vec.shape)
'''
# generate correlation matrix
corr = matrix @ matrix.T
shape = corr.shape
print(shape)
'''
'''Performing power iteration'''
'''
approx_vals = []
v0 = np.random.randn(shape[1])
v = v0

# ground truth eigenvalue
face_eigvals_truth, face_eigvecs_truth = la.eig(corr)
face_eigvals_truth = np.abs(face_eigvals_truth)

idx = face_eigvals_truth.argsort()[::-1]
face_eigvals_truth = face_eigvals_truth[idx]
face_eigvecs_truth = face_eigvecs_truth[:, idx]

max_face_eigvals_truth = face_eigvals_truth[0]
max_face_eigvecs_truth = face_eigvecs_truth[:, 0]

# normalize intial eigenvector guess
v = v / la.norm(v)
max_eigval_approx = v.T @ ( corr @ v ) # (1,m) * (m,m) * (m,1) = (1,1)
approx_vals.append(max_eigval_approx)

print('\nDominant eigenvalue: ', max_face_eigvals_truth)
print('\nStarting guess lambda: ', approx_vals[0])
print(' ')
'''
'''
for i in range(num_iterations):

    # calculating next eigenvector approximation
    v_next = corr @ v # (m,1)

    # calculate norm
    v_next_norm = la.norm(v_next, 2)

    # normalizing eigenvector approximation
    v = v_next / v_next_norm # (m,1)

    max_eigval_approx = v.T @ ( corr @ v ) # (1,m) * (m,m) * (m,1) = (1,1)
    approx_vals.append(max_eigval_approx)
    print(f'i = {i}, lambda = {max_eigval_approx}')

    # print(max_eigval_approx)
    # approx_vals_real.append(max_eigval_approx.real)
    # approx_vals_imag.append(max_eigval_approx.imag)
    # error_vecs.append(la.norm(np.abs(v) - np.abs(max_eigvec_truth), 2))
    # print(max_eigval_approx)
    # print(la.norm(v - max_eigvec_truth, 2))
    # error_vals.append(np.abs(max_eigval_approx) - np.abs(max_eigval_truth))

# print('\nGround truth max eigvec\n----------------------\n', max_eigvec_truth)
# print(f'\nApproximate max eigvec after {num_iterations} iterations\n----------------------\n', v)

# print('\nGround truth max eigval\n----------------------\n', max_eigval_truth)
# print(f'\nApproximate max eigval after {num_iterations} iterations\n----------------------\n', max_eigval_approx)

#
# # print(max_eig_approx)
#
# # X,Y = np.meshgrid(m,m)
# # plt.plot(error_vecs)
# # # plt.yscale('log')
# # plt.xlabel('Iteration')
# # plt.ylabel('Error')
# # plt.show()
#
# plt.scatter(approx_vals_real, approx_vals_imag)
# plt.show()
'''
'''
# print(approx_vals)
plt.plot(approx_vals - np.max(np.abs(face_eigvals_truth)))
# plt.yscale('log')
plt.title(f'Average faces, dominant eigenvalue approximation error: power iteration', fontsize=10)
plt.xlabel('Iteration', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.yscale('log')
plt.savefig('hw5q2_eigapprox.png', bbox_inches='tight')
plt.show()

# Perform SVD on vectorized images
u_cropped, s_cropped, v_cropped = la.svd(corr, full_matrices=False)
print('\nError: ', approx_vals - s_cropped[0])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(np.arange(0, len(s_cropped[:50])), s_cropped[:50], label='Cropped Images')
ax1.plot(0, approx_vals[-1], 'go', label='Eigenvalue Approximation')
plt.legend()
plt.title('s_cropped')
plt.savefig('s_cropped_hw5q2.png', bbox_inches='tight')
plt.show()

#  plot dominant eigenvector against leading mode
plt.plot(v, u_cropped[:,0])
plt.title('dominant eigenvector vs. leading SVD mode')
plt.savefig('leadingsvd_vs_domeigvec.png', bbox_inches='tight')
plt.show()
'''
# STAGE A
K = list(range(5,56, 10)) # number of random projections
U = []
Q = []
uapprox = []
A_estimate_list = []
# A_recon_list = []

for k in K:
    Omega = np.random.randn(matrix.shape[1], 5)
    Y = matrix @ Omega
    # print(Y.shape)
    #
    # plt.pcolor(Omega)
    # plt.show()

    Q, R = la.qr(Y, mode='economic')

    # STAGE B
    B = Q.T @ matrix
    U1, S, V = la.svd(B, full_matrices=False)
    U.append(U1)

    uapprox.append(Q @ U[-1])
    S = np.diag(S)
    A_estimate = uapprox[-1] @ S @ V
    # np.shape(A_estimate)

    A_estimate = A_estimate.reshape(2432, res[0], res[1])
    A_estimate_list.append(A_estimate[:3])
    A_recon = U1 @ S @ V
    print(A_recon.shape)
    A_recon = A_recon.reshape(5, res[0], res[1])
    fig, ax = plt.subplots(11,3)
    # plt.subplots_adjust(hspace=0.01, wspace=0.01)

# A_estimate_list = np.array(A_estimate_list).reshape(10, 3, res[0], res[1])

for i in range(10):
    for h in range(3):
        ax[i,h].imshow(A_estimate_list[i, h], cmap='gray')
        ax[i,h].axis('off')

for h in range(3):
    ax[10, h].imshow(A_recon[h], cmap='gray')
    ax[10, h].axis('off')
    plt.savefig('cropped_faces.png', bbox_inches='tight')

plt.show()
