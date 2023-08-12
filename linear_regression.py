import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from gradient_descent3 import GradientDescent
from sklearn.preprocessing import StandardScaler 

@dataclass
class Prediction:

	#read csv
	def get_data(self, path):
		dataset = pd.read_csv('data.csv')
		return dataset
	
	#Create a prompt
	def get_integer(self):
		while True:
			try:
				value = int(input("Enter the mileage of your car: "))
				return value
			except ValueError:
				print("Invalid input. Please enter an integer.")

	# add a column full of one and convert to numpy array
	def add_one_column(self, x):
		x = np.asarray(x)
		if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
			X = np.column_stack((np.ones(len(x)), x))
			return X
		else:
			print("ERROR: x Numpy Array")
			exit()
	
	def predict_(self, new_predict, thetas, path):
		dataset = self.get_data(path)
		x = dataset['km']
		x = x.values.reshape((-1, 1))
		scaler = StandardScaler()
		x_scaled = scaler.fit_transform(x)
		new_predict = np.array([new_predict]).reshape((-1, 1))
		new_predict = scaler.transform(new_predict)
		if isinstance(x, (int, float)):
			X = np.array([1, new_predict])
		else:
			X = self.add_one_column(new_predict)
			y_hat = np.matmul(X, thetas) 
		return y_hat

def main():
	path =  'data.csv'
	gd = GradientDescent()
	pr = Prediction()
	value = pr.get_integer()
	thetas = np.array([0, 0])
	thetas = gd.gradient_descent(path)
	new_y_hat = pr.predict_(value, thetas, path)
	print(f" prediction = {new_y_hat}")

if __name__ == "__main__":
    main()
