import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
import scipy.io as sio

m = 5
n = 3

# a = np.array([[1,1,0],[1,0,1],[0,1,1]])
# m = a.shape[0]
# n = a.shape[1]
# a = np.random.rand(m,n)
a = sio.loadmat('/home/billyh/Documents/AMATH 584: Linear Algebra/LA HW Code/hw3/mats2/a.mat')['A']
# print(a)
# a[:,-1] = a[:,0].copy()
print(np.linalg.cond(a))
q, r = la.qr(a, mode='economic', check_finite=False)


q_gs = np.zeros((q.shape))
r_gs = np.zeros((r.shape))
e_gs = np.zeros((a.shape))
u_gs = np.zeros((a.shape))
# #
u_gs[:,0] = a[:,0].copy()
e_gs[:,0] = np.divide(u_gs[:,0],la.norm(u_gs[:,0])).astype('float64')
q_gs[:,0] = e_gs[:,0]

for column in range(1, n):
    a_vec = a[:, column]
    u = a_vec.copy().astype('float64')

    for indx in range(column):
        u -= (np.matmul(a_vec, e_gs[:, indx]) * e_gs[:, indx]).astype('float64')

    u_gs[:, column] = u.copy()
    e_gs[:, column] = np.divide(u_gs[:, column], la.norm(u_gs[:, column]))
    q_gs[:, column] = e_gs[:, column].copy()

# r_gs = np.triu(a)
r_gs = np.matmul(q_gs.T, a)
# sio.savemat('r_gs_python.mat', {'r_gs': r_gs})
r_matlab = sio.loadmat('/home/billyh/Documents/AMATH 584: Linear Algebra/LA HW Code/hw3/mats3/r_qralg_matlab.mat')['r1']
print(q_gs)
# print(la.norm(r_gs))
for indx in range(q_gs.shape[1]):
    print(la.norm(q_gs[:,indx]))
# plt.matshow(r_gs)
# plt.show()
