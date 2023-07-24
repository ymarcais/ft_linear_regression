# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymarcais <ymarcais@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/12/13 15:18:06 by ymarcais          #+#    #+#              #
#    Updated: 2023/07/24 16:45:41 by ymarcais         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #pip install scikit-learn

class MyLinearRegression():

	#initialisation
	def __init__(self, thetas, alpha=[0.001, 0.000340], max_iter=50000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		
	# We need to add a column full of one
	def add_one_column(self, x):
		x = np.asarray(x)
		if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
			X = np.column_stack((np.ones(len(x)), x))
			return X
		else:
			print("ERROR: x Numpy Array")
			exit()
		
	# Gradient Calculation
	def simple_gradient(self, x, y):
		m = len(x)
		X = self.add_one_column(x)
		y_hat = self.predict_(x)
		cost = (y_hat - y)
		gradient = (1/(2 * m)) * np.dot(X.T, cost)
		return gradient
		
	# prediction: y_hat = ŷ = hθ(x)
	def predict_(self, x):
		X = self.add_one_column(x)
		y_hat = np.matmul(X, self.thetas)
		return y_hat

	# Gradient descent limitted with double alphas speed and maximum iteration and stoped to epsilon delta
	def gradient_descent(self, x, y, thetas, alpha, max_iter, epsilon=1e-5):
		i= 0
		m = len(x)
		prev_cost = 10.0
		alpha = self.alpha
		alpha = np.reshape(alpha, (2, 1))
		while i < max_iter:
			gradient = self.simple_gradient(x, y)
			thetas -= alpha * gradient
			i += 1
			y_hat = self.predict_(x)
			current_cost = (1 / (2 * m)) * np.sum((y_hat - y)**2)
			if current_cost - prev_cost < epsilon :
				print("Converged at iteration", i+1)
				break
			alpha *= 0.99999
		return thetas
		
	# double plot existing data and prediction
	def plot_(self, x, y, thetas, new_y_hat):
		plt.title("Prediction by Linear Regression")
		plt.plot(x, y, 'o', color='blue')
		plt.plot([50000], new_y_hat, 'o', color='red')
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
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	alpha = mlr.alpha
	max_iter = mlr.max_iter
	thetas = mlr.gradient_descent(x_scaled, y, thetas, alpha, max_iter)
	new_predict = np.array([50000]).reshape((-1, 1))
	new_predict = scaler.transform(new_predict)
	new_y_hat = mlr.predict_(new_predict)
	mlr.plot_( x, y, thetas, new_y_hat)

if __name__ == "__main__":
    main()
