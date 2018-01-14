#!/usr/bin/env python

import numpy as np
from perceptron import train
 
'''
for multi class, choose one as positive sample, others are negative samples
''' 

if __name__ == '__main__':
	D = 2
	w1 = [[-1,-1]] #
	w2 = [[0,0]]   #
	w3 = [[1,1]]  
	y2 = [-1]*len(w2)
	y3 = [-1]*len(w3)
	#choose w1 as positive samples
	y1 = [1] * len(w1)
	X = np.concatenate((w1,w2,w3),axis = 0)
	b = np.ones(len(X))
	b.shape = (len(b), 1)
	X = np.concatenate((X,b), axis = 1)
	Y = np.concatenate((y1,y2,y3),axis = 0)

	theta1 = np.random.normal(0,0.01, 1+D) # parameters
	train(X,Y,theta1)
	print 'theta1:', theta1
	#choose w2 as positive samples
	y1 = [-1]*len(w1)
	y2 = [1]*len(w2)
	Y = np.concatenate((y1,y2,y3),axis = 0)
	theta2 = np.random.normal(0,0.01, 1+D)
	train(X,Y,theta2)
	print 'theta2:', theta2
	#choose w3 as positive samples
	y2 = [-1]*len(w2)
	y3 = [1]*len(w3)
	Y = np.concatenate((y1,y2,y3), axis = 0)
	theta3 = np.random.normal(0,0.01, 1+D)
	train(X,Y,theta3)
	print 'theta3:', theta3
