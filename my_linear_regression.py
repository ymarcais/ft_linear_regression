# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymarcais <ymarcais@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/12/13 15:18:06 by ymarcais          #+#    #+#              #
#    Updated: 2023/06/20 18:16:18 by ymarcais         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #pip install scikit-learn

class MyLinearRegression():

	def __init__(self, thetas, alpha=[0.001, 0.000340], max_iter=50000):
		self.alpha = alpha
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
		
	def simple_gradient(self, x, y):
		m = len(x)
		X = self.add_one_column(x)
		y_hat = self.predict_(x)
		cost = (y_hat - y)
		print("cost", np.sum(cost))
		gradient = (1/(2 * m)) * np.dot(X.T, cost)
		return gradient
		
	def predict_(self, x):
		X = self.add_one_column(x)
		y_hat = np.matmul(X, self.thetas)
		return y_hat

	def fit_(self, x, y, thetas, alpha, max_iter, epsilon=1e-5):
		i= 0
		m = len(x)
		prev_cost = 10.0
		alpha = self.alpha
		alpha = np.reshape(alpha, (2, 1))
		while i < max_iter:
			gradient = self.simple_gradient(x, y)
			#print("gradient", gradient)
			thetas -= alpha * gradient
			i += 1
			#print("thetas: ", thetas)
			y_hat = self.predict_(x)
			current_cost = (1 / (2 * m)) * np.sum((y_hat - y)**2)
			if current_cost - prev_cost < epsilon :
				print("Converged at iteration", i+1)
				break
			alpha *= 0.99999
			#print("current cost: ", current_cost)
		return thetas
	
	def plot(self, x, y, thetas):
		plt.plot(x, y, 'o', color='blue')
		plt.plot(x, self.predict_(x), color='red')
		plt.show()


def main():
	dataset = pd.read_csv('data.csv')
	thetas = np.array([0, 0]).reshape((-1, 1))
	thetas = thetas.astype('float64')
	mlr = MyLinearRegression(thetas=thetas)
	x = dataset['km']
	x = x.values.reshape((-1, 1))
	y = dataset['price']
	y = y.values.reshape((-1, 1))
	m = len(x)
	#y_hat = mlr.predict_(x)
	#normalization of x
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	alpha = mlr.alpha
	max_iter = mlr.max_iter
	thetas = mlr.fit_(x_scaled, y, thetas, alpha, max_iter)
	print("thetas:", thetas)
	#mlr.plot(x_scaled, y, thetas)
	'''thetas[0] = 10000
	thetas[1] = -0.025
	y_hat = mlr.predict_(x)
	#theo_cost = (1 / (2 * m)) * np.sum((y_hat - y)**2)
	theo_cost = np.sum((y_hat - y))
	print("theo_cost =", theo_cost)'''
	new_predict = np.array([50000]).reshape((-1, 1))
	new_predict = scaler.transform(new_predict)
	new_y_hat = mlr.predict_(new_predict)
	print("new_predict", new_predict)
	print("new_y_hat", new_y_hat)
	plt.plot(x, y, 'o', color='blue')
	plt.plot([50000], new_y_hat, 'o', color='red')
	plt.show()

if __name__ == "__main__":
    main()


