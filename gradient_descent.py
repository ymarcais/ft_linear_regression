import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class GradientDescent:
    dataset: pd.DataFrame = field(default=None)
    learning_rate: 0.01
    
    # extract vector from dataset
    def set_vector(self, axe):
        self.dataset = pd.read_csv('data.csv')
        vector = self.dataset[axe]
        return vector

    # add a column full of one and convert to numpy array
    def add_one_column(self, x):
        x = self.set_vector('km')
        x = np.asarray(x)
        if isinstance(x, np.ndarray) or x.size != 0:
            X = np.column_stack((np.ones(len(x)), x))
            return X
        else:
            print("ERROR: x Numpy Array")
            exit()

    #y_hat = X * theta
    def predict_(self, x, theta):
        X = self.add_one_column(x)
        if isinstance(theta, np.ndarray or the.size != 0):
            y_hat = np.dot(X, theta)
            return y_hat
        else:
            print("ERROR: theta Numpy Array")
            exit()
    
    # tmp_theta0 = (1 / m) * sum 01 -> m-1 ((y_hat(i) - y(i))**2)
    def tmp_theta0(self, y, y_hat):
        m = len(y)
        t = np.transpose(y_hat - y)
        tmp_theta0 = learning_rate * (1 / m) * sum(np.dot(t, (y_hat - y)))
        return tmp_theta0
        



def main():
    gd = GradientDescent()
    '''x = gd.set_vector('km')
    y = gd.set_vector('price')
    X = gd.add_one_column(x)
    theta = np.random.randn(2)
    #y_hat = gd.predict_(x, theta)
    plt.scatter(x, y)
    plt.plot(sorted(x), y_hat, color='red')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()'''
    x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    loss = gd.loss_(x, y)

    print(loss)

    
if __name__ == "__main__":
    main()  
    






