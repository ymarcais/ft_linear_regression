import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler #pip install scikit-learn

# dataclass init instances
@dataclass
class GradientDescent:
	thetas: np.ndarray
	alpha: np.ndarray
	max_iter: int

	# add a column full of one and convert to numpy array
	def add_one_column(self, x):
		x = np.asarray(x)
		if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
			X = np.column_stack((np.ones(len(x)), x))
			return X
		else:
			print("ERROR: x Numpy Array")
			exit()

	#y prediction: y_hat = X * theta
	def predict_(self, x):
		X = self.add_one_column(x)
		y_hat = np.matmul(X, self.thetas)
		return y_hat
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i)))
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * 1 
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * x0(i) // rewrite 1 as x0(i) :
	#tmp_theta(1) = (1/m)sum((hθ(x(i) ) − y(i))) * x1(i)
	#Vectorisation:
		# hθ (x) = X' * θ    // hθ (x) = θ0 + θ1 x
		# tmp_theta(j) = (1 / m) * (X' * θ - y) * X'(j)
		# tmp_theta = (1 / m) * transpose(X') * (X' * θ - y) 
	def simple_gradient(self, x, y):
		m = len(x)
		X = self.add_one_column(x)
		y_hat = self.predict_(x)
		cost = (y_hat - y)
		#print("cost", np.sum(cost))
		gradient = (1/(2 * m)) * np.dot(X.T, cost)
		return gradient

	# gradient descent
	def fit_(self, x, y, thetas, alpha, max_iter, epsilon=1e-3):
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
			#print("current cost: ", current_cost)
			#print("prev_cost: ", prev_cost)
			if abs(current_cost - prev_cost) < epsilon :
				print("Converged at iteration", i+1)
				break
			alpha *= 0.99999
			prev_cost = current_cost
		return thetas

	def plot(self, x_scaled, x, y, thetas, new_y_hat):
		fig, axs = plt.subplots(1, 2)
		axs[0].plot(x_scaled, y, 'o', color='blue')
		axs[0].set_title("Linear Regression")
		axs[0].set_xlabel("km - x normalized")
		axs[0].set_ylabel("Price")
		axs[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[0].plot(x_scaled, self.predict_(x_scaled), color='red')

		axs[1].plot(x, y, 'o', color='blue')
		axs[1].plot([50000], new_y_hat, 'o', color='red')
		axs[1].set_title("Prediction")
		axs[1].set_xlabel("km")
		axs[1].set_ylabel("Price")
		axs[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

		plt.tight_layout()
		plt.show()

def main():
	dataset = pd.read_csv('data.csv')
	thetas = np.array([0, 0]).reshape((-1, 1))
	thetas = thetas.astype('float64')
	max_iter = 50000
	alpha = np.array([0.001, 0.000340])
	alpha = alpha.astype('float64')
	gd = GradientDescent(thetas, alpha, max_iter)
	x = dataset['km']
	x = x.values.reshape((-1, 1))
	y = dataset['price']
	y = y.values.reshape((-1, 1))
	m = len(x)
	#y_hat = gd.predict_(x)
	#normalization of x
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	max_iter = gd.max_iter
	thetas = gd.fit_(x_scaled, y, thetas, alpha, max_iter)
	#print("thetas:", thetas)
	new_predict = np.array([50000]).reshape((-1, 1))
	new_predict = scaler.transform(new_predict)
	new_y_hat = gd.predict_(new_predict)
	#print("new_predict", new_predict)
	#print("new_y_hat", new_y_hat)
	gd.plot(x_scaled, x, y, thetas, new_y_hat)

if __name__ == "__main__":
	main()







