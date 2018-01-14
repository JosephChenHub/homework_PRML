#!/usr/bin/env python

import numpy as np
'''
perceptron algorithm
1.hypothesis function: h_w(x) = sign(w'x+b)
sign(x) = 1, if x >= 0 otherwise sign(x) = -1
2.update parameters:
 w_j = w_j + lr*( y^i- h_w(x^i))*x_j^i
 
'''
max_iters = 100000
lr = 1
def hypothesis(x, theta):
	if np.dot(theta, x) >= 0:
		return 1
	else:
		return -1

def total_loss(x, y, theta):
	out = 0
	for i in range(len(x)):
		h = hypothesis(x[i], theta)
		if not h == y[i]: 
			out += 1
	return out

def train(X,Y,theta):
	loss = total_loss(X,Y,theta)
	print 'learning rate:', lr
	print 'initial theta:', theta
	print 'initial loss:', loss
	for it in range(max_iters):
		for i in range(len(X)): #a training set
			h = hypothesis(X[i], theta) 
			theta += lr*(Y[i] - h)*X[i]

		loss = total_loss(X,Y, theta)			 
		if __name__ == '__main__':
			print 'iter:', it
			print 'theta:', theta
			print 'loss:', loss
		if loss == 0:
			print 'optimization finished!'
			return
	if loss != 0:
		print 'not linearly separable!'
	return

if __name__ == '__main__':
	w1 = [[0,0,0],[1,0,0],[1,0,1],[1,1,0]] #label y = 1
	w2 = [[0,0,1],[0,1,1],[0,1,0],[1,1,1]] #label y = -1
	y1 = [1]*len(w1)
	y2 = [-1]*len(w2)
	X = np.concatenate((w1,w2),axis = 0)
	b = np.ones(len(X))
	b.shape = (len(b), 1)
	X = np.concatenate((X,b), axis = 1)
	Y = np.concatenate((y1,y2),axis = 0)
	W = np.random.normal(0,0.01,4)


	train(X,Y,W)
	print 'theta:', W
