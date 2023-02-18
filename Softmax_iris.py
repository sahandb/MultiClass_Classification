import numpy as np
import pandas as pd
#import pygame as pg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import math
import warnings
#warnings.filterwarnings('ignore')


#Split the data into 80% training and 20% testing
data_train = pd.read_csv('iris_Train_oneVsAll(120).csv')
data_test = pd.read_csv('iris_Test_oneVsAll(30).csv')




x = data_train[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']]
n = x.shape[1]
m = x.shape[0]

y = data_train['variety']
k = len(y.unique())
y =y.map({'Setosa':0,'Versicolor':1,'Virginica':2})

x[5] = np.ones(x.shape[0])


theta = np.empty((k,n+1))

alpha = 0.1
num_iter = 100



def phi(i,theta,x):  #i goes from 1 to k
    mat_theta = np.matrix(theta[i])
    mat_x = np.matrix(x)
    num = math.exp(np.dot(mat_theta,mat_x.T))
    den = 0
    for j in range(0,k):
        mat_theta_j = np.matrix(theta[j])
        den = den + math.exp(np.dot(mat_theta_j,mat_x.T))
    phi_i = num/den
    return phi_i

def indicator(a,b):
    if a == b: return 1
    else: return 0

def derivative_gradient(j,theta):
    sum = np.array([0 for i in range(0,n+1)])
    for i in range(0,m):
        p = indicator(y[i],j) - phi(j,theta,x.loc[i])
        sum = sum + (x.loc[i] *p)
    grad = -sum/m
    return grad



def gradient_descent(theta,alpha,num_iter):
    for j in range(0,k):
        for iter in range(num_iter):
            theta[j] = theta[j] - alpha * derivative_gradient(j,theta)
            print("class: ",j,'iterations: ',iter)
    return theta

def h_theta(x):
    x = np.matrix(x)
    h_matrix = np.empty((k,1))
    den = 0
    for j in range(0,k):
        den = den + math.exp(np.dot(theta_dash[j], x.T))
    for i in range(0,k):
        h_matrix[i] = math.exp(np.dot(theta_dash[i],x.T))
    h_matrix = h_matrix/den
    return h_matrix


theta_dash = gradient_descent(theta,alpha,num_iter)

print(theta_dash)

x_u = data_test[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']]
n = x_u.shape[1]
m = x_u.shape[0]

y_true = data_test['variety']
k = len(y_true.unique())
y_true =y_true.map({'Setosa':0,'Versicolor':1,'Virginica':2})
y_true.value_counts()

x_u[5] = np.ones(x_u.shape[0])

for index,row in x_u.iterrows():
    h_matrix = h_theta(row)
    prediction = int(np.where(h_matrix == h_matrix.max())[0])
    x_u.loc[index,'prediction'] = prediction


results = x_u
results['actual'] = y_true

print(results.head(30))



compare = results['prediction'] == results['actual']
correct = compare.value_counts()[1]
accuracy = correct/len(results)

print("acc is : ",accuracy * 100)







   

