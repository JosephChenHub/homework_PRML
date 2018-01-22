#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


#========= generate data =========#
Mu1 = np.array([1,2])
Sigma1 = np.array([[1,0.4],[0.4,0.8]])

Mu2 = np.array([5,1.5])
Sigma2 = np.array([[1,0.2],[0.2, 0.6]])

D = 2
N1 = 20
N2 = 20
x1 = np.random.multivariate_normal(Mu1, Sigma1, N1)
x2 = np.random.multivariate_normal(Mu2, Sigma2, N2)

#======= show data's distribution ==========#
plt.scatter(x1[:,0], x1[:,1], c = 'g', marker = 'o')
plt.scatter(x2[:,0], x2[:,1], c = 'b', marker = '^')
plt.show()

#======== Fisher's linear discriminat ======#
m1 = np.mean(x1, axis = 0)
m2 = np.mean(x2, axis = 0)
m1 = np.matrix(m1)
m2 = np.matrix(m2)
m1.shape = (D,1)
m2.shape = (D,1)
print(' x1.shape:{} x2.shape:{}'.format(x1.shape, x2.shape))

print(' m1:\n {} m2:\n {}'.format(m1,m2))
#between-class covariance
Sb = (m2 - m1)*(m2-m1).T
print(' between-class covariance:\n {}'.format(Sb))

#within-class covariance
Sw = np.matrix(np.zeros((D,D)) )
for i in range(N1):
	Sw += (x1[i,:] - m1)*(x1[i,:] - m1).T
for i in range(N2):
	Sw += (x2[i,:] - m2)*(x2[i,:] - m2).T

print(' within-class covariance:\n {}'.format(Sw))

w = np.linalg.inv(Sw)*(m2 - m1)
y0 = np.array(0.5*w.T*(m1+m2))
y0.shape = (1,)
print 'y0:', y0

print(' decision boundary: {}^Tx - {} = 0'.format(w, y0))

px1 = x1*w
px2 = x2*w

plt.scatter(x1[:,0], x1[:,1], c = 'g', marker = 'o')
plt.scatter(x2[:,0], x2[:,1], c = 'b', marker = '^')

w = np.array(w)
w.shape = (D,1)
t = np.linspace(0,6,100)
y1 = (y0 - w[0]*t)/w[1]
plt.plot(t, y1 , 'r', label = 'w^Tx - y0 = 0')

plt.savefig('fisher.eps')
plt.show()
