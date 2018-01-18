#!/usr/bin/env python

import numpy as np
import cvxpy as cvx #to solve convex problems
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from homework2.perceptron import train

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#====== generate test data =========#

Mu1 = np.array([1,3])
Sigma1 = np.array([[1,0.5],[0.5,1.2]])

Mu2 = np.array([5,1])
Sigma2 = np.array([[1,0.4],[0.4,0.8]])

N1 = 40
N2 = 30
x1 = np.random.multivariate_normal(Mu1,Sigma1, N1) #label 1
y1 = np.ones(N1)
x2 = np.random.multivariate_normal(Mu2,Sigma2, N2) #label -1
y2 = -np.ones(N2)

#show original data
plt.scatter(x1[:,0],x1[:,1],c='g', marker='o')
plt.scatter(x2[:,0],x2[:,1],c='b', marker='^')
plt.show()

#linear classifier

X = np.concatenate((x1,x2), axis = 0)
x3 = np.ones((N1+N2,1))
X = np.concatenate((X,x3) , axis = 1) #shape N,3
Y = np.concatenate((y1,y2), axis = 0) #shape N,1
W = np.random.normal(0, 0.01, 3)
train(X,Y,W)
print('theta:{}'.format(W))
t = np.linspace(0,10,100)
ft = -(W[2] + W[0]*t)/W[1]
plt.scatter(x1[:,0],x1[:,1],c='g', marker='o')
plt.scatter(x2[:,0],x2[:,1],c='b', marker='^')
plt.plot(t,ft, 'r')
#plt.savefig('linear_sp.eps')
plt.show()

#solve QP
W2 = cvx.Variable(3,1) 
objective = cvx.Minimize(0.5*cvx.sum_squares(W2))
constraints = []
for i in range(N1+N2):
	constraints.append(Y[i]*(W2.T*X[i,:]) >= 1)

prob = cvx.Problem(objective, constraints)
result = prob.solve()
print('problem status:{}'.format(prob.status))
W3 = np.array(W2.value.tolist())

print('optimal value:{}'.format(W3))


#plot
t = np.linspace(0,10,100)
ft = -(W[2] + W[0]*t)/W[1]
plt.scatter(x1[:,0],x1[:,1],c='g', marker='o')
plt.scatter(x2[:,0],x2[:,1],c='b', marker='^')
plt.plot(t,ft, 'r')
t = np.linspace(0,10,100)
ft = -(W3[2] + W3[0]*t)/W3[1]
plt.plot(t,ft,c='c', label = r'$w^Tx+b=0$')

ft1 = (1 - W3[2] - W3[0]*t)/W3[1]
plt.plot(t,ft1, '--', c='k',label=r'$w^Tx+b=1$')
ft2 = (-1 - W3[2] - W3[0]*t)/W3[1]
plt.plot(t, ft2, '--', c = 'm',label=r'$w^Tx+b=-1$')
plt.legend(loc='upper right')


plt.savefig('svm_p.eps')

plt.show()




