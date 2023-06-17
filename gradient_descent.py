import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class GradientDescent:
    #dataset: pd.DataFrame = field(default=None)
    alpha : float = 0.0005
    max_iter : int = 60000

    theta = np.array([2.0, 0.0])
    theta = theta.reshape((-1, 1))
    
    # add a column full of one and convert to numpy array
    def add_one_column(self, x):
        x = np.asarray(x)
        if isinstance(x, (np.ndarray, pd.DataFrame)) and x.size != 0:
            X = np.column_stack((np.ones(len(x)), x))
            return X
        else:
            print("ERROR: x Numpy Array")
            exit()

    #y_hat = X * theta
    def predict_(self, x):
        X = self.add_one_column(x)
        #print("X shape", np.shape(X))
        #print("X value", X)
        #print("theta value", theta)
        self.theta = self.theta
        #print(theta)
        y_hat = np.matmul(X, self.theta)
        #print("y_hat", y_hat)
        return y_hat
       
       
    #tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i)))
    #tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * 1 
    #tmp_theta(0) = (1/m)sum((hθ(x(i) ) − y(i))) * x0(i) // rewrite 1 as x0(i) :
    #tmp_theta(1) = (1/m)sum((hθ(x(i) ) − y(i))) * x1(i)
    #tmp_theta(j) = (1/m)sum((hθ(x(i) ) − y(i))) * xj(i)
    #Vectorisation:
        # hθ (x) = X' * θ    // hθ (x) = θ0 + θ1 x
        # tmp_theta(j) = (1 / m) * (X' * θ - y) * X'(j)
        # tmp_theta = (1 / m) * transpose(X') * (X' * θ - y) 
    def gradient(self, x, y):
        m = len(x)
        X = self.add_one_column(x)
        Xt = np.transpose(X)
        #print("Xt value", Xt)
        #print("Xt", np.shape(Xt))
        y_hat = self.predict_(x)
        #print("y_hat", np.shape(y_hat))
        cost = y_hat - y
        #print("y_hat value", y_hat)
        #print("y_hat", np.shape(y_hat))
        #print("y value", y)
        #print("cost", cost)
        #print("cost", np.shape(cost))
        grad = (1 / m) * np.dot(Xt, cost)
        #print("grad", grad)
        return grad

    #descent by iteration    
    def descent(self, x, y, alpha, max_iter):
        i = 0
        while i < max_iter:
            #plt.plot(x, self.predict_(x), color='red')
            grad = self.gradient(x, y)
            self.theta  = self.theta - alpha * grad
            #print("theta", np.shape(theta))
            print("theta value", self.theta)
            i += 1
        return self.theta

    '''def descent(self, x, y, alpha, max_iter):
        i = 0
        y_hat0 = 2**36
        X = self.add_one_column(x)
        grad = self.gradient(x, y)
        self.theta  = self.theta - alpha * grad
        y_hat1 = np.dot(X, self.theta)
        while np.any(y_hat1 < y_hat0):
            y_hat0 = y_hat1
            grad = self.gradient(x, y)
            self.theta = self.theta - alpha * grad
            print("theta value", self.theta)
            X = self.add_one_column(x)
            y_hat1 = np.dot(X, self.theta)
        return self.theta'''

    def plot(self, x, y, theta):
        plt.plot(x, y, 'o', color = 'blue')
        plt.plot(x, self.predict_(x), color='red')
        plt.show()
    

def main():
    dataset = pd.read_csv('data.csv')
    gd = GradientDescent()
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    #x = dataset['km']
    #x = x.values.reshape((-1, 1))
    #print("x", x)
    #y = dataset['price']
    #y = y.values.reshape((-1, 1))
    #print("y", y)
    X = gd.add_one_column(x)
    #theta = np.array([3, 2]).reshape((-1, 1))
    y_hat = gd.predict_(x)
    #print("y_hat", y_hat)
    #print("shape y_hat", np.shape(y_hat))
    #print("theta", np.shape(theta))
    alpha = gd.alpha
    max_iter = gd.max_iter
    grad = gd.gradient(x, y)
    #print("grad", np.shape(grad))
    #print("grad", np.size(grad))
    theta = gd.descent(x, y, alpha, max_iter)
    print("theta: ", theta)
    gd.plot(x, y, theta)

            
if __name__ == "__main__":
    main()  
    






