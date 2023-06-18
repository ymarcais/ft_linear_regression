# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymarcais <ymarcais@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/12/13 15:18:06 by ymarcais          #+#    #+#              #
#    Updated: 2023/06/18 20:33:11 by ymarcais         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #pip install scikit-learn



class MyLinearRegression():

	def __init__(self, thetas, alpha=[0.0, 0.0], max_iter=1000000):
		self.alpha = [0.0001, 0.0000000005]
		self.max_iter = max_iter
		self.thetas = thetas
		
	def add_one_column(self, x):
		x = np.asarray(x)
		if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
			X = np.column_stack((np.ones(len(x)), x))
			return X
		else:
			print("ERROR: x Numpy Array")
			exit()
		
	def simple_gradient( self, x, y, thetas):
		df = pd.DataFrame(x)
		shape = df.shape
		m = shape[0]
		v1 = [1] * m 
		v1 = np.reshape(v1, (m, 1))
		xp = np.concatenate((v1, x), axis = 1)
		xp2 = np.dot(xp, thetas)
		xpt = xp.transpose()
		sub = xp2 - y
		grad = (1 / m) *np.dot(xpt, sub)

		return grad
		
	def predict_(self, x):
		X = self.add_one_column(x)
		#print("X shape", np.shape(X))
		#print("X value", X)
		#print("theta value", theta)
		#self.theta = self.theta
		#print(theta)
		y_hat = np.matmul(X, self.thetas)
		#print("predict thetas:", self.thetas)
		#print("y_hat", y_hat)
		return y_hat

	def fit_(self, x, y, thetas, alpha, max_iter, epsilon=1e-6):
		i= 0
		m = len(x)
		prev_cost = 10.0
		alpha = self.alpha
		alpha = np.reshape(alpha, (2, 1))
		while i < max_iter:
			gradient = self.simple_gradient(x, y, thetas)
			thetas -= alpha * gradient
			i += 1
			print("thetas: ", thetas)
			y_hat = self.predict_(x)
			current_cost = (1 / 2 * m) * np.sum((y_hat - y)**2)
			# Check for convergence
			print("ecart cost =", abs(current_cost - prev_cost))
			if abs(current_cost - prev_cost) < epsilon:
				print("Converged at iteration", i+1)
				break
			prev_cost = current_cost
			alpha *= 0.99999
		print("current cost: ", current_cost)
		return thetas

	'''def predict_(self, x, thetas):
		m = x.shape[0]
		x = np.reshape(x, (m, 1))
		v1 = [1] * m
		v1 = np.reshape(v1, (m, 1))
		xp = np.concatenate((v1, x), axis = 1)
		y_hat = np.dot(xp, thetas)
		return y_hat'''
		
	
	
	def plot(self, x, y, thetas):
		plt.plot(x, y, 'o', color='blue')
		plt.plot(x, self.predict_(x), color='red')
		plt.show()


def main():
	dataset = pd.read_csv('data.csv')
	thetas = np.array([0.0, 0.0]).reshape((-1, 1))
	mlr = MyLinearRegression(thetas=thetas)
	
	x = dataset['km']
	x = x.values.reshape((-1, 1))
	#print("x shape", np.shape(x))
	y = dataset['price']
	y = y.values.reshape((-1, 1))
	#print("y shape", np.shape(y))
	
	# Feature scaling
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	y_scaled = scaler.fit_transform(y)
	'''mean_x = np.mean(x)
	std_x = np.std(x)
	x_scaled = (x - mean_x)  / std_x'''
	
	
	alpha = mlr.alpha
	max_iter = mlr.max_iter

	thetas = mlr.fit_(x_scaled, y, thetas, alpha, max_iter)
	print("thetas final: ", thetas)
	#thetas[0] = thetas[0] *1.5
	y_hat = mlr.predict_(x)
	current_cost = np.sum((y_hat - y))
	print("current_cost *1.5 =", current_cost)
	mlr.plot(x, y, thetas)

if __name__ == "__main__":
    main()


