import numpy as np
import pandas as pd
#import pygame as pg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Split the data into 80% training and 20% testing
data_train12 = pd.read_csv('iris_oneVsOne(class12_Train).csv')
data_test12 = pd.read_csv('iris_oneVsOne(class12_Test).csv')

data_train13 = pd.read_csv('iris_oneVsOne(class13_Train).csv')
data_test13 = pd.read_csv('iris_oneVsOne(class13_Test).csv')

data_train23 = pd.read_csv('iris_oneVsOne(class23_Train).csv')
data_test23 = pd.read_csv('iris_oneVsOne(class23_Test).csv')

#12
X_train12 = data_train12.iloc[:, :-1].values
Y_train12 = data_train12.iloc[:,4]

X_test12 = data_test12.iloc[:, :-1].values
Y_test12 = data_test12.iloc[:,4]

#13
X_train13 = data_train13.iloc[:, :-1].values
Y_train13 = data_train13.iloc[:,4]

X_test13 = data_test13.iloc[:, :-1].values
Y_test13 = data_test13.iloc[:,4]

#23
X_train23 = data_train23.iloc[:, :-1].values
Y_train23 = data_train23.iloc[:,4]

X_test23 = data_test23.iloc[:, :-1].values
Y_test23 = data_test23.iloc[:,4]


#1*train and test
X__train12 = np.c_[np.ones((len(X_train12),1)),X_train12]
X__test12 = np.c_[np.ones((len(X_test12),1)),X_test12]

X__train13 = np.c_[np.ones((len(X_train13),1)),X_train13]
X__test13 = np.c_[np.ones((len(X_test13),1)),X_test13]

X__train23 = np.c_[np.ones((len(X_train23),1)),X_train23]
X__test23 = np.c_[np.ones((len(X_test23),1)),X_test23]


####
Y_Train12_firstClassOneVsOne = np.where(Y_train12 == 'Setosa', 1, 0)

Y_Test12_firstClassOneVsOne = np.where(Y_test12 == 'Setosa', 1, 0)


########

Y_Train13_secondClassOneVsOne = np.where(Y_train13 == 'Virginica', 1, 0)


Y_Test13_secondClassOneVsOne = np.where(Y_test13 == 'Virginica', 1, 0)


##########
Y_Train23_thirdClassOneVsOne = np.where(Y_train23 == 'Versicolor', 1, 0)

Y_Test23_thirdClassOneVsOne = np.where(Y_test23 == 'Versicolor', 1, 0)



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

    return np.where(sigmoid(X,theta) >= 0.5, 1, 0)



# m = Number of training examples
# n = number of features
m, n = X__train12.shape

# initialize theta(weights) parameters to zeros
theta = np.zeros(1+n)
thetas = np.zeros((3,1+n))

# set learning rate to 0.01 and number of iterations to 500
alpha = 0.001
num_iter = 50000

cost = lrGradient(X__train12, Y_Train12_firstClassOneVsOne, theta, alpha, num_iter)
thetas[0,:] = theta 

theta = np.zeros(1+n)
cost = lrGradient(X__train13, Y_Train13_secondClassOneVsOne, theta, alpha, num_iter)
thetas[1,:] = theta 

theta = np.zeros(1+n)
cost = lrGradient(X__train23, Y_Train23_thirdClassOneVsOne, theta, alpha, num_iter)
thetas[2,:] = theta 


Y_predict_first = lrPredict(X__train12,thetas[0,:])
Y_predict_second = lrPredict(X__train13,thetas[1,:])
Y_predict_third = lrPredict(X__train23,thetas[2,:])

Y_predict_Test1 = lrPredict(X__test12,thetas[0,:])
Y_predict_Test2 = lrPredict(X__test13,thetas[1,:])
Y_predict_Test3 = lrPredict(X__test23,thetas[2,:])


print(10*'***','train12',10*'***')
print('y pred train 12\n',Y_predict_first)
print(10*'***','train13',10*'***')
print('y pred train 13\n',Y_predict_second)
print(10*'***','train23',10*'***')
print('y pred train 23\n',Y_predict_third)


print(10*'***','test12',10*'***')
print('y pred test with theta class 1\n',Y_predict_Test1)
print(10*'***','test13',10*'***')
print('y pred test with theta class 2\n',Y_predict_Test2)
print(10*'***','test23',10*'***')
print('y pred test with theta class 3\n',Y_predict_Test3)

    


