import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
import scipy.io as sio

x = list(np.arange(1.920, 2.080, 0.001))
# print(x)
# def square(a):
#     return a*a

# test = map(square, x)
# print(list(test))
def p_xa(p):
    return p**9 - 18*(p**8) + 144*(p**7) - 672*(p**6) + 2016*(p**5) - \
    4032*(p**4) + 5376*(p**3) - 4608*(p**2) + 2304*(p) - 512
def p_xb(p):
    return (p - 2)**9

# part a
y1 = map(p_xa, x)
y2 = map(p_xb, x)
# print(list(y))

plt.plot(x, list(y1), label='RHS', color='b')
plt.plot(x, list(y2), label='LHS', color='g', ls='--')
plt.title('HW3 Q2', fontsize='xx-large')
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.legend(prop = {'size': '15'})
plt.savefig(f'hw3part2.png', bbox_inches='tight')
plt.show()
