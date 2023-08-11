import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler #pip install scikit-learn

# dataclass init instances
@dataclass
class GradientDescent:
	alpha: np.ndarray
	thetas: list[float] = field(default_factory=lambda: [0, 0])
	max_iter: int = 50000

	def get_data(self, path):
		dataset = pd.read_csv('data.csv')
		return dataset

	# add a column full of one and convert to numpy array
	def add_one_column(self, x):
		x = np.asarray(x)
		if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
			X = np.column_stack((np.ones(len(x)), x))
			return X
		else:
			print("ERROR: x Numpy Array")
			exit()

	#X Scaled before prediction : y prediction: y_hat = X * theta
	def predict_(self, x):
		X = self.add_one_column(x)
		#X = StandardScaler().fit_transform(X)
		y_hat = np.matmul(X, self.thetas) 
		#print(f"y hat shape: {y_hat.shape}")
		print(f"X: {X}")
		print(f"thetas: {self.thetas}")
		return y_hat

	#ŷ = hθ(x)
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i)))
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * 1 
	#tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * x0(i) // rewrite 1 as x0(i) :
	#tmp_theta(1) = (1/m)sum((hθ(x(i) ) − y(i))) * x1(i)
	#Vectorisation:
		# hθ (x) = X' * θ    // hθ (x) = θ0 + θ1 x
		# tmp_theta(j) = (1 / m) * (X' * θ - y) * X'(j)
		# tmp_theta = (1 / m) * transpose(X') * (X' * θ - y) 
	def simple_gradient(self, x, y):
		X = self.add_one_column(x)
		m = len(x)
		y_hat = self.predict_(x)
		cost = (y_hat - y)
		#print(f"cost : {cost}")
		gradient = (1/(2 * m)) * np.dot(X.T, cost)
		return gradient

	# gradient descent
	def gradient_descent(self, path, alpha, epsilon=1e-3):
		i= 0
		prev_cost = 10.0
		dataset = self.get_data(path)
		x = dataset['km']
		x = x.values.reshape((-1, 1))
		m = len(x)
		y = dataset['price']
		y = y.values.reshape((-1, 1))
		alpha = self.alpha
		alpha = np.reshape(alpha, (2, 1))
		self.thetas = np.array([0, 0]).reshape((-1, 1))
		self.thetas = self.thetas.astype('float64')
		while i < self.max_iter:
			gradient = self.simple_gradient(x, y)
			self.thetas -= alpha * gradient
			#print(f"thetas : {self.thetas}")
			#print(f"alpha : {alpha}")
			#print(f"gradient : {gradient}")
			i += 1
			y_hat = self.predict_(x)
			current_cost = (1 / (2 * m)) * np.sum((y_hat - y)**2)
			#print(f"current cost  = {current_cost}")
			if abs(current_cost - prev_cost) < epsilon :
				print("Converged at iteration", i+1)
				break
			alpha *= 0.99999
			prev_cost = current_cost
		return self.thetas

	#Chart with data and regression line
	def plot(self, path, new_predict):
		fig, axs = plt.subplots(1, 2)
		dataset = self.get_data(path)
		x = dataset['km']
		x = x.values.reshape((-1, 1))
		print(f"x shape: {x.shape}")
		x_scaled = StandardScaler().fit_transform(x)
		print(f"x_scale shape: {x_scaled.shape}")
		y = dataset['price']
		y = y.values.reshape((-1, 1))
		print(f"y shape: {y.shape}")

		self.thetas = self.gradient_descent(path, self.alpha)
		new_predict = np.array([new_predict]).reshape((-1, 1))
		new_predict = StandardScaler().transform(new_predict)
		new_y_hat = gd.predict_(new_predict)
		
		axs[0].plot(x_scaled, y, 'o', color='blue')
		axs[0].set_title("Linear Regression")
		axs[0].set_xlabel("km - x normalized")
		axs[0].set_ylabel("Price")
		axs[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x_scaled:,.0f}'))
		axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x_scaled:,.0f}'))
		axs[0].plot(x_scaled, self.predict_(path), color='red')

		axs[1].plot(x_scaled, y, 'o', color='blue')
		axs[1].plot([new_predict], new_y_hat, 'o', color='red')
		axs[1].set_title("Prediction")
		axs[1].set_xlabel("km")
		axs[1].set_ylabel("Price")
		axs[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x_scaled:,.0f}'))
		axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x_scaled:,.0f}'))

		plt.tight_layout()
		plt.show()

def main():
	path = 'data.csv'
	#scaler = StandardScaler()
	alpha = np.array([0.001, 0.000340])
	alpha = alpha.astype('float16')
	new_predict = 5000
	gd = GradientDescent(alpha = alpha)
	#gd.gradient_descent(path, alpha)
	#thetas = gd.gradient_descent(path, alpha)
	#new_predict = np.array([50000]).reshape((-1, 1))
	#new_predict = scaler.fit_transform(new_predict)
	#new_y_hat = gd.predict_(new_predict)
	#gd.plot(path, new_predict)
	gd.predict_([5000])

'''def main():
	path = 'data.csv'
	alpha = np.array([0.001, 0.000340])
	gd = GradientDescent(alpha = alpha)
	dataset = gd.get_data(path)
	x = dataset['km']
	x = x.values.reshape((-1, 1))
	y = dataset['price']
	y = y.values.reshape((-1, 1))
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	thetas = gd.gradient_descent(x_scaled, y, alpha)
	new_predict = np.array([50000]).reshape((-1, 1))
	new_predict = scaler.transform(new_predict)
	new_y_hat = gd.predict_(new_predict)
	gd.plot(x_scaled, x, y, new_y_hat)'''

if __name__ == "__main__":
	main()







