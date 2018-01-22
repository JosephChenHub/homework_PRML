#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

N1 = 20
N2 = 30
Mu1 = np.array([-1,-1])
Sigma1 = np.array([[1,0.4],[0.4,0.5]])

Mu2 = np.array([3,3])
Sigma2 = np.array([[2,0.5],[0.5, 0.8]])

x1 = np.random.multivariate_normal(Mu1, Sigma1, N1)
x2 = np.random.multivariate_normal(Mu2, Sigma2, N2)

#show 
plt.scatter(x1[:,0], x1[:,1], c = 'g', marker = 'o')
plt.scatter(x2[:,0], x2[:,1], c = 'b', marker = '^')

plt.show()

mu1 = np.array([0.,3.])
mu2 = np.array([3.,0.])

last_mu1 = np.array([0.,0.])
last_mu2 = np.array([0.,0.])
x = np.concatenate((x1,x2), axis = 0)
y = np.zeros(N1+N2)

while True:
	if abs(np.dot(mu1 - last_mu1, mu1 - last_mu1)) < 1e-6 and abs(np.dot(mu2 - last_mu2, mu2 - last_mu2)) < 1e-6:
		break
	last_mu1 = mu1
	last_mu2 = mu2
	#E step
	for i in range(N1+N2):
		dis1 = np.dot(x[i] - mu1, x[i] - mu1)
		dis2 = np.dot(x[i] - mu2, x[i] - mu2)
		if dis1 < dis2:
			y[i] = 1
		else:
			y[i] = -1
	#M step
	mu1 = np.array([0.,0.])
	mu2 = np.array([0.,0.])
	sum1 = 0
	sum2 = 0
	t1 = []
	t2 = []
	for i in range(N1+N2):
		if y[i] == 1:
			mu1 += x[i]
			sum1 += 1
			t1.append(x[i])
		elif y[i] == -1:
			mu2 += x[i]	
			sum2 += 1	
			t2.append(x[i])
	mu1 = mu1*1./sum1
	mu2 = mu2*1./sum2
	
	print('mu1:{} mu2 :{}'.format(mu1, mu2))
	t1 = np.array(t1)
	t2 = np.array(t2)
		
	plt.scatter(t1[:,0], t1[:,1], c = 'g', marker = 'o')
	plt.scatter(t2[:,0], t2[:,1], c = 'b', marker = 'o')
	plt.show()
