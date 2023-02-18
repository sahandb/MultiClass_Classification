import numpy as np
import pandas as pd
#import pygame as pg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Split the data into 80% training and 20% testing
data_train = pd.read_csv('iris_Train_oneVsAll(120).csv')
data_test = pd.read_csv('iris_Test_oneVsAll(30).csv')

#X = data.iloc[:, :2]
X_train = data_train.iloc[:, :-1].values
Y_train = data_train.iloc[:,4]

X_test = data_test.iloc[:, :-1].values
Y_test = data_test.iloc[:,4]



Y_Train_firstClassOneVsAll = np.where(Y_train == 'Setosa', 1, 0)
Y_Train_secondClassOneVsAll = np.where(Y_train == 'Versicolor', 1, 0)
Y_Train_thirdClassOneVsAll = np.where(Y_train == 'Virginica', 1, 0)

Y_Test_firstClassOneVsAll = np.where(Y_test == 'Setosa', 1, 0)
Y_Test_secondClassOneVsAll = np.where(Y_test == 'Versicolor', 1, 0)
Y_Test_thirdClassOneVsAll = np.where(Y_test == 'Virginica', 1, 0)



X__train = np.c_[np.ones((len(X_train),1)),X_train]
X__test = np.c_[np.ones((len(X_test),1)),X_test]



def sigmoid(X, theta):
    
    z = np.dot(X, theta[1:]) + theta[0]
    
    return 1.0 / ( 1.0 + np.exp(-z))

def lrCostFunction(y, hx):
  
    # compute cost for given theta parameters
    j = -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))
    
    return j

def lrGradient(X, y, theta, alpha, num_iter):
    # empty list to store the value of the cost function over number of iterations
    cost = []
    
    for i in range(num_iter):
        # call sigmoid function 
        hx = sigmoid(X, theta)
        # calculate error
        error = hx - y
        # calculate gradient
        grad = X.T.dot(error)
        # update values in theta
        theta[0] = theta[0] - alpha * error.sum()
        theta[1:] = theta[1:] - alpha * grad
        
        cost.append(lrCostFunction(y, hx))
        
    return cost





def lrPredict(X,theta):
    #return np.where(sigmoid(X,theta) >= 0.5, 1, 0)
    return sigmoid(X,theta)



# m = Number of training examples
# n = number of features
m, n = X__train.shape

# initialize theta(weights) parameters to zeros
theta = np.zeros(1+n)
thetas = np.zeros((3,1+n))

# set learning rate to 0.01 and number of iterations to 500
alpha = 0.001
num_iter = 50000

cost = lrGradient(X__train, Y_Train_firstClassOneVsAll, theta, alpha, num_iter)
thetas[0,:] = theta 
theta = np.zeros(1+n)
cost = lrGradient(X__train, Y_Train_secondClassOneVsAll, theta, alpha, num_iter)
thetas[1,:] = theta 
theta = np.zeros(1+n)
cost = lrGradient(X__train, Y_Train_thirdClassOneVsAll, theta, alpha, num_iter)
thetas[2,:] = theta 


Y_predict_first = lrPredict(X__train,thetas[0,:])
Y_predict_second = lrPredict(X__train,thetas[1,:])
Y_predict_third = lrPredict(X__train,thetas[2,:])

Y_predict_Test1 = lrPredict(X__test,thetas[0,:])
Y_predict_Test2 = lrPredict(X__test,thetas[1,:])
Y_predict_Test3 = lrPredict(X__test,thetas[2,:])


print(10*'***','train',10*'***')
print('y pred train 1\n',Y_predict_first)
print(10*'***','train',10*'***')
print('y pred train 2\n',Y_predict_second)
print(10*'***','train',10*'***')
print('y pred train 3\n',Y_predict_third)


print(10*'***','test',10*'***')
print('y pred test with theta class 1\n',Y_predict_Test1)
print(10*'***','test',10*'***')
print('y pred test with theta class 2\n',Y_predict_Test2)
print(10*'***','test',10*'***')
print('y pred test with theta class 3\n',Y_predict_Test3)

    

