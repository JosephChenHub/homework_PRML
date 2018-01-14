#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#raw data
x1 = np.matrix([[0,0,0],[2,0,0],[2,0,1],[1,2,0]])
x2 = np.matrix([[0,0,1],[0,1,0],[0,-2,1],[1,1,-2]])

x1 = x1.T  #column vectors
x2 = x2.T 

X = np.concatenate((x1,x2), axis = 1)
print('Raw data:\n {}'.format(X))

mean = np.mean(X, axis = 1)
print('Mean value:{}'.format(mean))
Z = X - mean
print('substract the mean:\n{}'.format(Z))
R = (Z*Z.T)/4/2

print('R:{}'.format(R))

a,b = np.linalg.eig(R) #eigenvalues and eigenvectors
print('eigenvalues:{}'.format(a))
print('eigenvectors:\n{}'.format(b))

#choose a[0],a[1]
t = b[:,:2]
t2_x1 = t.T*x1
t2_x2 = t.T*x2
print('two dimension w1:{} \n w2:{}'.format(t2_x1,t2_x2))

#choose a[0]
t = b[:,0]
t1_x1 = t.T*x1
t1_x2 = t.T*x2
print('one dimension w1:{} \n w2:{}'.format(t1_x1,t1_x2))

#plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

px, py, pz = x1[0],x1[1],x1[2]
ax.scatter(px, py, pz, c = 'r', marker = 'o')
px, py, pz = x2[0],x2[1],x2[2]
ax.scatter(px, py, pz, c = 'b', marker = '^')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('raw.eps')
plt.show()
#two

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(t2_x1[0,:].tolist(), t2_x1[1,:].tolist(), c = 'r', marker = 'o')
ax.scatter(t2_x2[0,:].tolist(), t2_x2[1,:].tolist(), c = 'b', marker = '^')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.savefig('two.eps')
plt.show()
#one
t1_x1 = np.array(t1_x1)
t1_x2 = np.array(t1_x2)

fig = plt.figure()
ax = fig.add_subplot(111)
y1 = np.zeros([1,t1_x1.shape[1]])
y2 = np.zeros([1,t1_x2.shape[1]])
print(t1_x1)
print(y1)

ax.scatter(t1_x1, y1, c = 'r', marker = 'o')
ax.scatter(t1_x2, y2, c = 'b', marker = '^')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.savefig('one.eps')
plt.show()
