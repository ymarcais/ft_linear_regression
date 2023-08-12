import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler #pip install scikit-learnfrom line

# dataclass init instances
@dataclass
class GradientDescent:
	alpha: np.ndarray = np.array([0.001, 0.000340], dtype=np.float16)
	thetas: list[float] = field(default_factory=lambda: [0, 0])
	max_iter: int = 50000

	#read csv
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
		y_hat = np.matmul(X, self.thetas) 
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
		gradient = (1/(1 * m)) * np.dot(X.T, cost)
		return gradient

	# gradient descent
	def gradient_descent(self, path, epsilon=1e-3):
		i= 0
		prev_cost = 10.0
		dataset = self.get_data(path)
		x = dataset['km']
		x = x.values.reshape((-1, 1))
		scaler = StandardScaler()
		x_scaled = scaler.fit_transform(x)
		m = len(x_scaled)
		y = dataset['price']
		y = y.values.reshape((-1, 1))
		alpha = self.alpha
		alpha = np.reshape(alpha, (2, 1))
		self.thetas = np.array([0, 0]).reshape((-1, 1))
		self.thetas = self.thetas.astype('float64')
		while i < self.max_iter:
			gradient = self.simple_gradient(x_scaled, y)
			self.thetas -= alpha * gradient
			print(f"thetas : {self.thetas}")
			i += 1
			y_hat = self.predict_(x_scaled)
			current_cost = (1 / (2 * m)) * np.sum((y_hat - y)**2)
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
		#x_scaled = StandardScaler().fit_transform(x)
		#print(f"x_scale shape: {x_scaled.shape}")
		y = dataset['price']
		y = y.values.reshape((-1, 1))
		scaler = StandardScaler()
		x_scaled = scaler.fit_transform(x)
		self.thetas = self.gradient_descent(path)
		new_predict = np.array([new_predict]).reshape((-1, 1))
		new_predict = scaler.transform(new_predict)
		new_y_hat = self.predict_(new_predict)
		
		axs[0].plot(x_scaled, y, 'o', color='blue')
		axs[0].set_title("Linear Regression")
		axs[0].set_xlabel("km - x normalized")
		axs[0].set_ylabel("Price")
		axs[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[0].plot(x_scaled, self.predict_(x_scaled), color='red')

		axs[1].plot(x_scaled, y, 'o', color='blue')
		axs[1].plot(new_predict, new_y_hat, 'o', color='red')
		axs[1].set_title(f"Price Prediction = {new_y_hat[0][0]:,.2f} Euros" )
		axs[1].set_xlabel("km - x normalized")
		axs[1].set_ylabel("Price")
		axs[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
		axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

		plt.tight_layout()
		plt.show()

		return new_y_hat

def main():
	path = 'data.csv'
	new_predict = 50000
	gd = GradientDescent()
	gd.plot(path, new_predict)
	
if __name__ == "__main__":
	main()







